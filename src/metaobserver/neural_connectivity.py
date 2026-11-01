"""GPT-2 module, parameter, activation and attention tracing prototype."""
from __future__ import annotations
import argparse, json, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from torch import nn

def _summary(t: torch.Tensor) -> dict[str, Any]:
    x=t.detach(); f=x.float()
    return {"shape":list(x.shape),"dtype":str(x.dtype),"device":str(x.device),
            "numel":x.numel(),"mean":f.mean().item() if f.numel() else 0.0,
            "std":f.std(unbiased=False).item() if f.numel() else 0.0,
            "min":f.min().item() if f.numel() else 0.0,
            "max":f.max().item() if f.numel() else 0.0}

def inspect_gpt2(model: nn.Module, include_parameter_prefix: int=0) -> dict[str,Any]:
    modules=[]
    for name,mod in model.named_modules():
        if not name: continue
        modules.append({"id":f"module:{name}","name":name,
            "module_type":type(mod).__name__,
            "children":[f"module:{name}.{n}" for n,_ in mod.named_children()],
            "direct_parameters":[f"parameter:{name}.{n}" for n,_ in mod.named_parameters(recurse=False)]})
    params=[]
    for name,p in model.named_parameters():
        item={"id":f"parameter:{name}","name":name,"shape":list(p.shape),
              "dtype":str(p.dtype),"numel":p.numel(),"requires_grad":p.requires_grad,
              "statistics":_summary(p)}
        if include_parameter_prefix>0:
            item["values_prefix"]=p.detach().flatten()[:include_parameter_prefix].cpu().tolist()
        params.append(item)
    cfg=getattr(model,"config",None)
    return {"model_type":getattr(cfg,"model_type",type(model).__name__),
            "model_class":type(model).__name__,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "modules":modules,"parameters":params,
            "config":cfg.to_dict() if cfg is not None else {}}

def build_dependency_graph(model_map: dict[str,Any]) -> dict[str,Any]:
    nodes=[]; edges=[]
    for mod in model_map["modules"]:
        nodes.append({"id":mod["id"],"kind":"module","label":mod["name"],
                      "module_type":mod["module_type"]})
        edges.extend({"source":mod["id"],"target":child,"relation":"contains"}
                     for child in mod["children"])
        edges.extend({"source":mod["id"],"target":param,"relation":"owns_parameter"}
                     for param in mod["direct_parameters"])
    nodes.extend({"id":p["id"],"kind":"parameter","label":p["name"],"shape":p["shape"]}
                 for p in model_map["parameters"])
    return {"graph_type":"structural_module_parameter_graph",
            "edge_semantics":"Module containment and parameter ownership; not inferred neuron wires.",
            "nodes":nodes,"edges":edges}

class GPT2TraceRecorder:
    def __init__(self,model:nn.Module,tokenizer:Any,top_k:int=20):
        self.model=model; self.tokenizer=tokenizer; self.top_k=top_k; self.handles=[]

    def _remove_hooks(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def trace(self,text:str,output_dir:str|Path,max_new_tokens:int=0)->dict[str,Any]:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        self._remove_hooks(); self.model.eval()
        device=next(self.model.parameters()).device
        ids=self.tokenizer(text,return_tensors="pt")["input_ids"].to(device)
        tokens=self.tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
        captured={"module_outputs":{},"attention_modules":{},"mlp_outputs":{}}
        def hook(bucket,name):
            def record(_m,ins,result):
                value=result[0] if isinstance(result,tuple) and result else result
                if isinstance(value,torch.Tensor):
                    captured[bucket][name]={"input":_summary(ins[0]) if ins and isinstance(ins[0],torch.Tensor) else None,
                                            "output":_summary(value)}
            return record
        for name,mod in self.model.named_modules():
            if name.endswith(".attn"):
                self.handles.append(mod.register_forward_hook(hook("attention_modules",name)))
            elif name.endswith(".mlp"):
                self.handles.append(mod.register_forward_hook(hook("mlp_outputs",name)))
            elif name in {"transformer.wte","transformer.wpe","transformer.ln_f"}:
                self.handles.append(mod.register_forward_hook(hook("module_outputs",name)))
        try:
            with torch.inference_mode():
                result=self.model(input_ids=ids,output_attentions=True,
                                  output_hidden_states=True,use_cache=False,return_dict=True)
                logits=result.logits.detach().float()
                probs=torch.softmax(logits[0,-1],dim=-1)
                vals,idx=torch.topk(probs,min(self.top_k,probs.numel()))
                top=[{"token_id":int(i),"token":self.tokenizer.decode([int(i)]),
                      "probability":float(v)} for v,i in zip(vals.cpu(),idx.cpu())]
                hidden=[]
                for i,h in enumerate(result.hidden_states or ()):
                    item=_summary(h); item["stage"]="embedding_output" if i==0 else f"block_output:{i-1}"
                    if h.numel()<=250000: item["token_position_values"]=h.detach().float().cpu().tolist()
                    hidden.append(item)
                attentions=[]
                for i,a in enumerate(result.attentions or ()):
                    entry={"layer":i,"shape":list(a.shape),"head_count":int(a.shape[1]),
                           "query_length":int(a.shape[2]),"key_length":int(a.shape[3]),
                           "summary":_summary(a)}
                    if a.numel()<=1000000: entry["values"]=a.detach().float().cpu().tolist()
                    attentions.append(entry)
                generated=[]; context=ids.clone()
                for step in range(max_new_tokens):
                    step_result=self.model(input_ids=context,output_attentions=False,
                                           output_hidden_states=False,use_cache=False,return_dict=True)
                    step_probs=torch.softmax(step_result.logits[0,-1].float(),dim=-1)
                    token=int(torch.argmax(step_probs).item())
                    generated.append({"step":step,"context_length":int(context.shape[1]),
                                      "selected_token_id":token,
                                      "selected_token":self.tokenizer.decode([token]),
                                      "selected_probability":float(step_probs[token].item())})
                    context=torch.cat([context,torch.tensor([[token]],device=device)],dim=1)
        finally:
            self._remove_hooks()
        model_map=inspect_gpt2(self.model)
        graph=build_dependency_graph(model_map)
        stages=["tokenization","token_embeddings","positional_embeddings"]+[
            f"transformer_block:{i}" for i in range(max(0,len(hidden)-1))]+[
            "final_layer_norm","language_model_head","logits"]
        exec_nodes=[{"id":f"execution:{i}:{s}","kind":"execution_stage","label":s}
                    for i,s in enumerate(stages)]
        exec_edges=[{"source":exec_nodes[i]["id"],"target":exec_nodes[i+1]["id"],
                     "relation":"sequential_execution_order",
                     "note":"Observed stage order; not a full tensor-dependency proof."}
                    for i in range(len(exec_nodes)-1)]
        report={"schema_version":"0.1.0",
          "experiment_id":datetime.now(timezone.utc).strftime("gpt2_%Y%m%dT%H%M%S_%fZ"),
          "created_at":datetime.now(timezone.utc).isoformat(),
          "input":{"text":text,"input_ids":ids[0].cpu().tolist(),"tokens":tokens,
                   "sequence_length":int(ids.shape[1])},
          "model":model_map,"structural_graph":graph,
          "runtime_trace":{"module_outputs":captured["module_outputs"],
            "attention_modules":captured["attention_modules"],"mlp_outputs":captured["mlp_outputs"],
            "hidden_states":hidden,"attention":attentions,"logits":_summary(logits),
            "last_position_top_tokens":top,"generation_steps":generated,
            "execution_order_graph":{"nodes":exec_nodes,"edges":exec_edges}},
          "metadata":{"device":str(device),"torch_version":torch.__version__,
            "python_version":platform.python_version(),
            "attention_note":"Attention probabilities describe token-position routing weights; not semantic or causal explanations.",
            "graph_note":"Structural edges are containment/parameter ownership. Execution-order edges are not fine-grained tensor dependency edges."}}
        report_path=out/"trace_report.json"; graph_path=out/"model_graph.json"
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        graph_path.write_text(json.dumps(graph,indent=2),encoding="utf-8")
        return {"report_path":str(report_path),"graph_path":str(graph_path),
                "experiment_id":report["experiment_id"],"parameter_count":model_map["parameter_count"],
                "module_count":len(model_map["modules"]),"attention_layers_captured":len(attentions),
                "hidden_state_stages":len(hidden),"top_predictions":top[:5]}

def main():
    parser=argparse.ArgumentParser(description="Brahma MetaCore GPT-2 neural connectivity trace")
    parser.add_argument("--model",default="openai-community/gpt2")
    parser.add_argument("--text",default="What is an apple?")
    parser.add_argument("--output",default="reports/neural_connectivity/gpt2_demo")
    parser.add_argument("--max-new-tokens",type=int,default=8)
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install Transformers: python -m pip install transformers") from exc
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else "cpu" if args.device=="auto" else args.device)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,attn_implementation="eager").to(device)
    print(json.dumps(GPT2TraceRecorder(model,tokenizer).trace(
        args.text,args.output,args.max_new_tokens),indent=2))

if __name__=="__main__": main()
