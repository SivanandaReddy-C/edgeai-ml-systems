import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(0,d_model,2)*
            (-math.log(10000.0)/d_model)
        )

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer("pe",pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1)]

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,Q,K,V):
        d_k=Q.size(-1)
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(d_k)
        attention_weights=torch.softmax(scores,dim=-1)
        output=torch.matmul(attention_weights,V)

        return output,attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        assert d_model % num_heads==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        self.attention=ScaledDotProductAttention()

        self.W_o=nn.Linear(d_model,d_model)

    def split_heads(self,x):
        batch_size,seq_len,d_model=x.size()
        x=x.view(batch_size,seq_len,self.num_heads,self.d_k)
        x=x.transpose(1,2)
        return x
    
    def forward(self,Q,K,V):
        batch_size=Q.size(0)

        Q=self.W_q(Q)
        K=self.W_k(K)
        V=self.W_v(V)
 
        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)
 
        output,attention_weights=self.attention(Q,K,V)
 
        output=output.transpose(1,2)
  
        output=output.contiguous().view(batch_size,-1,self.d_model)
   
        output=self.W_o(output)
  
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()

        self.fc1=nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(d_ff,d_model)
    
    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()

        self.mha=MultiHeadAttention(d_model,num_heads)
        self.ffn=PositionwiseFeedForward(d_model,d_ff)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

    def forward(self,x):
        #Multi-head attention
        attn_output=self.mha(x,x,x)

        #Residual connection + LayerNorm
        x=self.norm1(x+attn_output)

        # Feed forward network
        ffn_output=self.ffn(x)

        #Residual connection + LayerNorm
        x=self.norm2(x+ffn_output)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers):
        super().__init__()

        self.layers=nn.ModuleList([TransformerEncoderLayer(d_model,num_heads,d_ff)
                                    for _ in range(num_layers)])
        
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)

        return x

class TransformerClassifier(nn.Module):
    def __init__(self,d_model=64,num_heads=8,d_ff=256,num_layers=2,num_classes=10):
        super().__init__()
        self.input_projection=nn.Linear(28,d_model)
        self.positional_encoding=PositionalEncoding(d_model)
        self.encoder=TransformerEncoder(d_model,num_heads,d_ff,num_layers)
        self.classifier=nn.Linear(d_model,num_classes)
    
    def forward(self,x):
        #x:[B,1,28,28]
        x=x.squeeze(1)      #[B,28,28]
        x=self.input_projection(x)  #[B,28,d_model]
        x=self.positional_encoding(x)
        x=self.encoder(x)
        x=x.mean(dim=1) #global average pooling
        x=self.classifier(x)

        return x

if __name__=="__main__":
    model=TransformerClassifier()
    x=torch.randn(2,1,28,28)
    out=model(x)
    print(out.shape)


   