require 'torch'
require 'paths'
require 'image'
require 'hdf5'

local NewMovingMNISTDataset = torch.class('NewMovingMNISTLoader')

if torch.getmetatable('dataLoader') == nil then
   torch.class('dataLoader')
end

function NewMovingMNISTDataset:__init(opt, data_type)
  -- Arguments:
  --   data_type (string): 'train', 'val', 'test', or 'long'
  local data
  self.opt = opt or {}
  -- Get path to videos file
  local dataPath
  if data_type == 'train' then
    dataPath = opt.dataRoot .. '/' .. opt.sliceName .. '_videos.h5'
  else
    dataPath = opt.dataRoot .. '/' .. opt.sliceName .. '_' .. data_type .. '_videos.h5'
  end
  local dataFile = hdf5.open(dataPath, 'r')
  data = dataFile:read('/data'):all()
  data = data:permute(2, 1, 3, 4)
  self.data = torch.Tensor(data:size())
  self.data:copy(data)
  self.N = self.data:size(1)
end

function NewMovingMNISTDataset:size()
  return self.N 
end 

function NewMovingMNISTDataset:normalize()
  self.data:div(255)
end

local dirs = {4, -3, -2, -1, 1, 2, 3, 4}

function NewMovingMNISTDataset:getSequence(x)
  local t = x:size(1)
  -- Get random video
  local idx = math.random(self.N)
  local frames = self.data[{ idx, {1, t}, {}, {} }]
  -- Populate tensor with video data
  x:copy(frames)
end

function NewMovingMNISTDataset:getBatch(n, T)
  local xx = torch.Tensor(T, unpack(self.opt.geometry))
  local x = {}
  for t=1,T do
    x[t] = torch.Tensor(n, unpack(self.opt.geometry))
  end
  for i = 1,n do
    self:getSequence(xx)
    for t=1,T do
      x[t][i]:copy(xx[t])
    end
  end 
  return x
end 

function NewMovingMNISTDataset:plotSeq(fname)
  print('plotting sequence: ' .. fname)
  local to_plot = {}
  local t = self.opt.T or 20 
  local n = 20
  local x = self:getBatch(n, t)
  for i = 1,n do
    for j = 1,t do
      table.insert(to_plot, x[j][i])
    end
  end 
  for i=1,#to_plot do
    to_plot[i][{ {}, {}, 1}]:fill(1)
    to_plot[i][{ {}, {}, 64}]:fill(1)
    to_plot[i][{ {}, 1, {}}]:fill(1)
    to_plot[i][{ {}, 64, {}}]:fill(1)
  end
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=t})
end

function NewMovingMNISTDataset:plot()
  local savedir = self.opt.save  .. '/data/'
  os.execute('mkdir -p ' .. savedir)
  self:plotSeq(savedir .. '/seq.png')
end

trainLoader = NewMovingMNISTLoader(opt or opt_t, 'train')
trainLoader:normalize()
valLoader = NewMovingMNISTLoader(opt or opt_t, 'val')
valLoader:normalize()
