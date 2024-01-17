import torch
from torch import nn 
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gap
from torch_geometric.nn import GATConv
from dataset_process import data


class EmbeddingBlock(nn.Module):
    def __init__(self, inchannels, embedding):
        super(EmbeddingBlock, self).__init__()
        
        self.embedding = nn.Linear(inchannels, embedding)
        self.conv1 = GATConv(embedding, embedding, heads= 12, concat=False)

    def forward(self, x, edge_index):

        embedding = torch.relu(self.embedding(x))

        y = self.conv1(embedding, edge_index)

        return torch.relu(embedding + y)

class MultiAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(MultiAttentionBlock, self).__init__()
        
        self.conv1 = GATConv(channels, channels, heads= 12, concat=False)

    def forward(self, x, edge_index):

        y = self.conv1(x, edge_index)

        return torch.relu(x + y)

class MLPBlock(nn.Module):
    def __init__(self, channels):
        super(MLPBlock, self).__init__()
        double_channels = int(channels*2)

        self.fcc1 = nn.Linear(channels, double_channels)
        self.fcc2 = nn.Linear(double_channels, channels)


    def forward(self, x):

        y = torch.relu(self.fcc1(x))
        y = self.fcc2(y)

        return torch.relu(x + y)

class EncoderBlock(nn.Module):
    def __init__(self, channels):
        super(EncoderBlock, self).__init__()

        self.multiattention = MultiAttentionBlock(channels)
        self.MLPblock = MLPBlock(channels)

    def forward(self, x, edge_index):

        y = self.multiattention(x, edge_index)
        y = self.MLPblock(y)
        return y

class DecoderBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DecoderBlock, self).__init__()
        self.conv = GATConv(inchannels, outchannels, heads = 12, concat= False)
        self.MLPblock = MLPBlock(outchannels)

    def forward(self, x, edge_index):

        y = torch.relu(self.conv(x, edge_index))
        y = self.MLPblock(y)
        return y

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        torch.manual_seed(3407)
        #-----------------encoder block-------------------
        self.conv1 = GATConv(data.num_features, 256, heads= 12, concat=False)
        self.encoderblock1 = EncoderBlock(256)
        self.encoderblock2 = EncoderBlock(256)
        self.encoderblock3 = EncoderBlock(256)

        #-----------------decoder block-------------------

        
        self.decoderblock1 = DecoderBlock(256, 128)
        self.decoderblock2 = MLPBlock(128)
        self.decoderblock3 = DecoderBlock(128, 64)
        self.decoderblock4 = MLPBlock(64)
        
        self.regression = nn.Linear(in_features = 64, out_features = 7)

    def forward(self, x , edge_index, batch_index, mask_edge):
        
        hidden = torch.relu(self.conv1(x, edge_index))
        hidden = self.encoderblock1(hidden, edge_index)
        hidden = self.encoderblock2(hidden, edge_index)
        hidden = self.encoderblock3(hidden, edge_index)

        x = gmp(hidden , batch_index)  
        x  = self.decoderblock1(x, mask_edge)
        x  = self.decoderblock2(x)
        x  = self.decoderblock3(x, mask_edge)
        x  = self.decoderblock4(x)
        y = self.regression(x)

        return y, hidden
        

if __name__ == '__main__':

    model = GAT()
    print(model)