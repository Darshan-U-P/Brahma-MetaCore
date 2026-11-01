import torch
import torch.nn as nn
from src.metaobserver.neural_connectivity import GPT2TraceRecorder, build_dependency_graph, inspect_gpt2

class TinyConfig:
    model_type="tiny-gpt2"
    def to_dict(self): return {"model_type":self.model_type}

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.config=TinyConfig()
        self.transformer=nn.Module()
        self.transformer.wte=nn.Embedding(32,8)
        self.transformer.wpe=nn.Embedding(16,8)
        self.transformer.ln_f=nn.LayerNorm(8)
        self.transformer.h=nn.ModuleList([nn.ModuleDict({
            "attn":nn.Linear(8,8),
            "mlp":nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))})])
        self.lm_head=nn.Linear(8,32,bias=False)
    def forward(self,input_ids,output_attentions=True,output_hidden_states=True,
                use_cache=False,return_dict=True):
        pos=torch.arange(input_ids.shape[1],device=input_ids.device).unsqueeze(0)
        x=self.transformer.wte(input_ids)+self.transformer.wpe(pos)
        for block in self.transformer.h: x=x+block["attn"](x); x=x+block["mlp"](x)
        x=self.transformer.ln_f(x)
        class Result: pass
        r=Result(); r.logits=self.lm_head(x)
        r.hidden_states=(x,) if output_hidden_states else None; r.attentions=None
        return r

class TinyTokenizer:
    def __call__(self,text,return_tensors="pt"): return {"input_ids":torch.tensor([[1,2,3]])}
    def convert_ids_to_tokens(self,ids): return [f"tok_{i}" for i in ids]
    def decode(self,ids): return f"tok_{ids[0]}"

def test_inventory_has_parameter_and_module_data():
    m=TinyModel(); report=inspect_gpt2(m)
    assert report["parameter_count"]==sum(p.numel() for p in m.parameters())
    assert any(x["name"]=="transformer.wte" for x in report["modules"])
    assert any(x["name"]=="transformer.wte.weight" for x in report["parameters"])

def test_graph_records_structural_edges():
    graph=build_dependency_graph(inspect_gpt2(TinyModel()))
    assert graph["nodes"]
    assert any(e["relation"]=="owns_parameter" for e in graph["edges"])
    assert any(e["relation"]=="contains" for e in graph["edges"])

def test_trace_serializes_and_cleans_up_hooks(tmp_path):
    m=TinyModel(); before={n:len(x._forward_hooks) for n,x in m.named_modules()}
    result=GPT2TraceRecorder(m,TinyTokenizer()).trace("What is an apple?",tmp_path)
    assert (tmp_path/"trace_report.json").exists()
    assert (tmp_path/"model_graph.json").exists()
    assert result["parameter_count"]>0
    for name,module in m.named_modules(): assert len(module._forward_hooks)==before[name]
