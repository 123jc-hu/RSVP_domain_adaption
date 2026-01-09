import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import pywt
from torchinfo import summary
from Utils.config import load_config


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, max_len, d_model=128, dropout=0.5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],  requires_grad=False).to(x.device)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_sizeh: int = 64, patch_sizew: int = 5, emb_size: int = 128, img_size1: int = 64, img_size2:int = 250):
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew
        super().__init__()
        self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, emb_size, kernel_size=(self.patch_sizeh,self.patch_sizew), stride=(self.patch_sizeh,self.patch_sizew),padding=(0,0)),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh)), emb_size))
        self.nonpara = PositionalEncoding(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh)))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x += self.positions
        #x = self.nonpara(x)
        return x


class Mutihead_Attention(nn.Module):
    def __init__(self,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)
        #self.v = self.k

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def generate_mask(self,dim,score):
        device = score.device
        thre = torch.mean(score,dim=-1).to(device) 
        thre = torch.unsqueeze(thre, 3)
        vec = torch.ones((1,dim)).to(device)
        thre = torch.matmul(thre,vec)  
        cha = score - thre
        one_vec = torch.ones_like(cha).to(device)
        zero_vec = torch.zeros_like(cha).to(device)
        mask = torch.where(cha > 0, zero_vec, one_vec).to(device)
        return mask==1

    def forward(self,x,y,requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(y).reshape(-1,y.shape[0],y.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        #print("Attention V shape : {}".format(V.shape))
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(attention_score.size()[3],attention_score)
            attention_score = attention_score.masked_fill(mask==True,value=float("-inf"))
        attention_score = F.softmax(attention_score,dim=-1)
        output = torch.matmul(attention_score,V).reshape(x.shape[0],x.shape[1],-1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output


class Feed_Forward(nn.Module):  #output
    def __init__(self,input_dim=128,hidden_dim=128*4,channel=4,kernel_size=32):
        super(Feed_Forward, self).__init__()
        F1 = channel
        self.conv1 = nn.Conv2d(1, F1, (kernel_size, 1), bias = False, stride = (1,1))  #Conv2d #F1*4*8
        self.dropout = nn.Dropout(0.5)
        self.gelu = GELU()

    def forward(self,x):  
        output = self.gelu(self.conv1(x.unsqueeze(1)))
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.num_flat_features(output))
        return output
        
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features
    

class Feed_Forward1(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Feed_Forward1, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)
        self.gelu = GELU()

    def forward(self,x):
        output = self.gelu((self.L1(x)))
        output = self.L2(output)
        return output
    

class Add_Norm(nn.Module):
    def __init__(self, size = 128):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(size)

    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)
        out = self.norm(x)
        return out


class Encoder(nn.Module):
    def __init__(self,dim_seq,dim_fea,n_heads,hidden):
        super(Encoder, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(self.dim_fea,self.dim_k,self.dim_v,self.n_heads)
        self.feed_forward = Feed_Forward1(self.dim_fea,self.hidden)
        self.add_norm = Add_Norm(self.dim_fea)

    def forward(self,x): 
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.add_norm(output,self.feed_forward)
        return output


class Decoder(nn.Module):
    def __init__(self,dim_seq,dim_fea,n_heads,hidden):
        super(Decoder, self).__init__()
        self.dim_seq = dim_seq 
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden
        
        self.muti_atten = Mutihead_Attention(self.dim_fea,self.dim_k,self.dim_v,self.n_heads)
        self.feed_forward = Feed_Forward1(self.dim_fea,self.hidden)
        self.add_norm = Add_Norm(dim_fea)

    def forward(self,q,v):
        output = self.add_norm(q,self.muti_atten,y=v,requires_mask=True)
        output = self.add_norm(output,self.feed_forward)
        output = output + q
        return output


class Cross_modal(nn.Module):
    def __init__(self, dim_seq=1*32, dim_fea=128, n_heads=4, hidden=128*4):
        super(Cross_modal, self).__init__()
        self.cross1 = Decoder(dim_seq, dim_fea, n_heads, hidden)
        self.cross2 = Decoder(dim_seq, dim_fea, n_heads, hidden)
        self.fc1 = nn.Linear(2*dim_fea,dim_fea)

    def forward(self,target,f1):
        re = self.cross1(target,f1)
        return re
        

class Cross_modalto(nn.Module):
    def __init__(self,dim_seq=4*32,dim_fea=128, n_heads=4, hidden=128*4):
        super(Cross_modalto, self).__init__()
        self.dim_seq = dim_seq
        self.long = 32
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(self.dim_fea,self.dim_k,self.dim_v,self.n_heads)
        self.add_norm = Add_Norm(dim_fea)

    def forward(self,q,v):
        output = self.add_norm(q,self.muti_atten,y=v,requires_mask=True)
        output = output + q
        return output
    

class Transformer_layer(nn.Module):
    def __init__(self, dmodel=128, num_heads=4, num_tokens=50):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder(num_tokens,dmodel,num_heads,dmodel*4)

    def forward(self,x):  
        encoder_output = self.encoder(x) + x
        return encoder_output


class Attention_score(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.proj = nn.Linear(d_model, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def forward(self, x_t, x_f):
        x_t = self.proj(x_t)
        x_f = self.proj(x_f)

        Att_t = torch.matmul(x_t,x_t.permute(0,2,1)) * self.norm_fact
        Att_f = torch.matmul(x_f,x_f.permute(0,2,1)) * self.norm_fact
        Att_tf = torch.matmul(x_t,x_f.permute(0,2,1)) * self.norm_fact
        Att_ft = torch.matmul(x_f,x_t.permute(0,2,1)) * self.norm_fact

        Att_t = F.softmax(Att_t,dim=-1)*(1-torch.eye(Att_t.shape[-1],device=x_t.device))
        Att_f = F.softmax(Att_f,dim=-1)*(1-torch.eye(Att_f.shape[-1],device=x_f.device))
        Att_tf = F.softmax(Att_tf,dim=-1)*(1-torch.eye(Att_tf.shape[-1],device=x_t.device))
        Att_ft = F.softmax(Att_ft,dim=-1)*(1-torch.eye(Att_ft.shape[-1],device=x_f.device))

        score_t = torch.sum(Att_t,dim=1).unsqueeze(2)
        score_f = torch.sum(Att_f,dim=1).unsqueeze(2)
        score_tf = torch.sum(Att_tf,dim=1).unsqueeze(2) #t对f的关注度
        score_ft = torch.sum(Att_ft,dim=1).unsqueeze(2)

        score_t = repeat(score_t, 'n l () -> n l d', d=x_t.shape[-1])
        score_f = repeat(score_f, 'n l () -> n l d', d=x_f.shape[-1])
        score_tf = repeat(score_tf, 'n l () -> n l d', d=x_t.shape[-1])
        score_ft = repeat(score_ft, 'n l () -> n l d', d=x_f.shape[-1])

        return score_t, score_f, score_tf, score_ft

    
class tokenchange(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.score_att = Attention_score(d_model=d_model)
        self.spe = nn.Linear(d_model,d_model)
        self.tem = nn.Linear(d_model,d_model)

    def generate_mask(self, score, time=True):
        '''
        score.shape: [N, L, D]
        '''
        device = score.device
        length = score.shape[1]

        theta, _ = torch.topk(score[:,:,0], k=int(0.5*length), dim=-1, largest=time)
        theta = (theta[:,-1].unsqueeze(1)).unsqueeze(2)
        theta = repeat(theta, 'n () d -> n l d', l=length)
        theta = repeat(theta, 'n l () -> n l d', d=self.d_model)
        cha = score - theta
        one_vec = torch.ones_like(cha).to(device)
        zero_vec = torch.zeros_like(cha).to(device)
        if time:
            mask = torch.where(cha > 0, one_vec, zero_vec).to(device) 
        else:
            mask = torch.where(cha > 0, zero_vec, one_vec).to(device)  
        return mask==1
    
    def generate_mask1(self, score, score_cross, time=True):
        '''
        score.shape: [N, L, D]
        '''
        device = score.device
        length = score.shape[1]

        cha = score - score_cross
        one_vec = torch.ones_like(cha).to(device)
        zero_vec = torch.zeros_like(cha).to(device)
        if time:
            mask = torch.where(cha > 0, one_vec, zero_vec).to(device)
        else:
            mask = torch.where(cha > 0, zero_vec, one_vec).to(device) 
        return mask==1

    def forward(self, x_t, x_f):

        score_t, score_f, score_tf, score_ft = self.score_att(x_t, x_f)

        mask_t = self.generate_mask(score_t,False) 
        mask_f = self.generate_mask(score_f,False)
        mask_tf = self.generate_mask(score_tf,True) 
        mask_ft = self.generate_mask(score_ft,True) 

        mask_t1 = mask_t*(~mask_f)
        mask_f1 = mask_f*(~mask_t)
        x_t1 = x_t*(~mask_t1) + x_f*mask_t1 
        x_f1 = x_f*(~mask_f1) + x_t*mask_f1

        return 0.5*(x_t1+x_t), 0.5*(x_f1+x_f)


class Interaction(nn.Module):
    def __init__(self, d_model, n_heads, h, w):
        super().__init__()
        self.t = Cross_modal(dim_seq=h*w, dim_fea=d_model, n_heads=n_heads, hidden=d_model*4)
        self.f = Cross_modal(dim_seq=h*w, dim_fea=d_model, n_heads=n_heads, hidden=d_model*4)
        self.change = tokenchange(d_model=d_model)
        self.muti_attent = Mutihead_Attention(d_model, d_model//n_heads, d_model//n_heads, n_heads)
        self.add_normt = Add_Norm(size=d_model)
        self.muti_attenf = Mutihead_Attention(d_model, d_model//n_heads, d_model//n_heads, n_heads)
        self.add_normf = Add_Norm(size=d_model)

    def forward(self, x):
        x_t, x_f = x[0], x[1]
        x_t2 = self.t(x_t,x_f)
        x_f2 = self.f(x_f,x_t)
        x_t2, x_f2 = self.change(x_t2,x_f2)
        x_t2 = self.add_normt(x_t2,self.muti_attent,y=x_t2)
        x_f2 = self.add_normf(x_f2,self.muti_attenf,y=x_f2)

        return [x_t2+x_t, x_f2+x_f]
    

class Adapter(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        channel = 16
        self.aggregate = Feed_Forward(d_model,d_model*4,channel=channel)
        self.fc = nn.Linear(channel*3*(d_model//16),2)

    def forward(self, x):
        x = self.aggregate(x)
        output = self.fc(x)

        return output
    

class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]
        self.patch_sizew = 4
        self.patch_sizeh = self.n_channels
        self.N = 1
        self.d_model = 32
        self.n_heads = 1

        ######################################################### encoder specifics ###############################################################
        self.embedding_t = PatchEmbedding(in_channels=1, patch_sizeh=self.patch_sizeh, patch_sizew=self.patch_sizew, emb_size=self.d_model, img_size1=config["n_channels"], img_size2=config["fs"])
        self.embedding_f = PatchEmbedding(in_channels=20, patch_sizeh=self.patch_sizeh, patch_sizew=self.patch_sizew, emb_size=self.d_model, img_size1=config["n_channels"], img_size2=config["fs"])
        self.norm = nn.LayerNorm(self.fs)
        self.norm1 = nn.LayerNorm(self.d_model)

        #Encoder
        self.model_t = nn.Sequential(*[Transformer_layer(dmodel=self.d_model, num_heads=self.n_heads, num_tokens=self.fs//self.patch_sizew) for _ in range(self.N)])
        self.model_f = nn.Sequential(*[Transformer_layer(dmodel=self.d_model, num_heads=self.n_heads, num_tokens=self.fs//self.patch_sizew) for _ in range(self.N)])
        self.fc1 = nn.Linear(self.d_model*2,self.d_model)
        self.model_last = Cross_modalto(dim_seq=4*self.fs//self.patch_sizew, dim_fea=self.d_model, n_heads=self.n_heads, hidden=self.d_model*4)
        
        #cross-modal
        self.interaction = nn.Sequential(*[Interaction(d_model=self.d_model, n_heads=self.n_heads, h=1, w=self.fs//self.patch_sizew) for _ in range(1)])
        # -------------------------------------------------------------------------

        ########################################################## Classifier #######################################################################
        self.aggregate = Feed_Forward(input_dim=self.d_model, hidden_dim=self.d_model*4, channel=4, kernel_size=self.fs//self.patch_sizew)
        self.aggregate_con = Feed_Forward(input_dim=self.d_model, hidden_dim=self.d_model*4, channel=4, kernel_size=self.fs//self.patch_sizew)
        self.fc = nn.Linear(4*self.d_model, 2)

        self.adapter = Adapter()


    def forward_encoder(self,raw,fre):
        x_t1 = self.embedding_t(raw)
        x_f1 = self.embedding_f(self.norm(fre))

        x_t = self.model_t(x_t1)
        x_f = self.model_t(x_f1)

        #cross-modal
        x = self.interaction([x_t, x_f])
        x_t1, x_f1 = x[0], x[1]

        x_t2 = x_t + x_t1
        x_f2 = x_f + x_f1

        x = torch.cat((x_t2,x_f2),axis=1)
        output = self.model_last(F.relu(self.fc1(torch.cat((x_t2,x_f2),axis=-1))),x)

        return output, [x_t2, x_f2]
    
    def forward_classifier(self, x):
        x = self.aggregate(x)
        output = self.fc(x)

        return output
    
    def forward_contrastive(self, x):
        xt_con = self.aggregate_con(x[0])
        xf_con = self.aggregate_con(x[1])

        return xt_con, xf_con

    def forward(self, raw):
        fre = eeg_to_dwt(raw, wavelet='mexh')  # Convert EEG to DWT
        latent_cls, latent_con = self.forward_encoder(raw, fre)
        output = self.forward_classifier(latent_cls)
        # outout_adapt = self.adapter(latent_cls)
        # output = output + outout_adapt
        t_con, f_con = self.forward_contrastive(latent_con)

        return output, t_con, f_con


def eeg_to_dwt(eeg, wavelet='mexh'):
    # eeg:(N, 1, C, T)
    eeg = eeg.squeeze(1)  # (N, C, T)
    num_samples = eeg.shape[0]
    eeg_dwt = []
    for i in range(num_samples):
        eeg_dwt.append(Wave(eeg[i]))  # (scale, C, T)
    eeg_dwt = np.stack(eeg_dwt, axis=0)  # (N, scale, C, T)
    eeg_dwt = torch.tensor(eeg_dwt, dtype=torch.float32)  # Convert to tensor
    return eeg_dwt.to(eeg.device)  # (N, scale, C, T)

def Wave(Raw):
    Raw = Raw.cpu()
    Raw = Raw.data.numpy()
    Raw = np.float16(np.squeeze(Raw))

    wavename = 'mexh'
    totalscal = Raw.shape[-1]//2  
    sr = Raw.shape[-1]
    Fc = pywt.central_frequency(wavename)               
    cparam = 2*Fc*totalscal      
    scales = cparam / np.arange(totalscal, 1, -1) 
    scales = scales[scales.shape[0]-20:]

    wav_matrix, _ = pywt.cwt(Raw, scales, wavename, 1/sr)
    wav_matrix = abs(wav_matrix)
    return wav_matrix


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状
