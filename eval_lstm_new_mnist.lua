require 'torch'  
require 'nn'  
require 'nngraph'  
require 'cudnn'  
require 'cunn'  
require 'optim'  
require 'pl'  
require 'paths'  
require 'image'  
require 'utils'
succ, debugger = pcall(require,'fb.debugger')

-- parse command-line options  
opt = lapp[[  
  -b,--batchSize     (default 8)               batch size  
  -g,--gpu           (default 0)                 gpu to use  
  --name             (default 'default')         checkpoint name
  --dataRoot         (default 'data/new_mnist')  data root directory
  --epochSize        (default 1000)             number of samples per epoch  
  --imageSize        (default 64)                size of image
  --dataset          (default new_moving_mnist)               dataset
  --normalize                                    if set normalize predicted pose vectors to have unit norm
  --rnnSize          (default 256)
  --rnnLayers        (default 2)
  --nThreads         (default 0)                 number of dataloading threads
  --dataPool         (default 200)
  --dataWarmup       (default 10)
  --nPast            (default 10)                number of frames to condition on.  
  --nFuture          (default 5)                 number of frames to predict.
  --nFutureLong      (default 490)               number of frames to predict long-term.
  --printEvery       (default 100)               Print stats after this many batches
  --plotEvery        (default 1000)              Plot images after this many batches
  --testEvery        (default 50)                Evaluate after this many batches
  --modelSliceName   (string)                    Name of the new MNIST slice the desired model was trained on
  --dataSliceName    (string)                    Name of the new MNIST slice to evaluate on
  --modelRootFmt     (default 'logs/new_moving_mnist/%s')
  --lstmModelFmt     (default 'logs/new_moving_mnist/%s/lstm/%s/model_best.t7')
]]

opt.sliceName = opt.dataSliceName
opt.modelRoot = (opt.modelRootFmt):format(opt.modelSliceName)
local saveTestImagesRoot = ('results/MNIST/images/data=%s/model=%s'):format(opt.dataSliceName, opt.modelSliceName)
local saveLongImagesRoot = ('results/MNIST/images/data=%s_long/model=%s'):format(opt.dataSliceName, opt.modelSliceName)

os.execute('mkdir -p ' .. saveTestImagesRoot)
os.execute('mkdir -p ' .. saveLongImagesRoot)

-- setup some stuff
torch.setnumthreads(1)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.gpu + 1)
print('<gpu> using device ' .. opt.gpu)

-- torch.manualSeed(opt.seed)
-- cutorch.manualSeed(opt.seed)
-- math.randomseed(opt.seed)

opt.modelPath = opt.modelRoot .. '/model_best.t7'
local nets = torch.load(opt.modelPath)

opt.nShare = nets.opt.nShare
opt.contentDim = nets.opt.contentDim
opt.poseDim = nets.opt.poseDim
-- opt.batchSize = nets.opt.batchSize
opt.geometry = nets.opt.geometry
opt.imageSize = nets.opt.imageSize 
opt.movingDigits = nets.opt.movingDigits


-- require 'data.data'
require 'data.new_moving_mnist'
testLoader = NewMovingMNISTLoader(opt or opt_t, 'test')
testLoader:normalize()
longLoader = NewMovingMNISTLoader(opt or opt_t, 'long')
longLoader:normalize()

local netEC = nets['netEC']
local netEP = nets['netEP']
local netD = nets['netD']
netEP:cuda()
netEC:cuda()

-- if netD is nil, then decoder built into netEC because unet architecture
if netD then
  netD:cuda()
  opt.unet = false
else
  print('found unet model')
  opt.unet = true
end
print(opt)

netEC:training()
netEP:training()

-- TODO: Properly load model weights
require 'models.lstm'
-- lstm = makeLSTM()


local x_content = {}
for i=1, opt.nShare do
  x_content[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

function squeeze_all(input)
  for i=1, #input do
    input[i] = torch.squeeze(input[i])
  end
  return input
end

function get_reps(x_seq, T)
  for i=1, opt.nShare do
    x_content[i]:copy(x_seq[i])
  end

  local x = {}
  for i=1,T do
    x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
    x[i]:copy(x_seq[i]) 
  end
  x[T+1] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))

  local pose_reps = {}
  for i=1,T+1 do 
    pose_reps[i] = netEP:forward(x[i]):clone()
  end
  squeeze_all(pose_reps)
  local content_rep
  if opt.unet then
    content_rep = netEC:forward({x_content, pose_reps[1]})[2] -- pose rep doesnt matter
  else
    content_rep = netEC:forward(x_content)
  end
  content_rep = torch.squeeze(content_rep)

  return content_rep, pose_reps
end


function addBorder(image, color)
  local numChannels = image:size(1)
  local height = image:size(2)
  local width = image:size(3)
  local ret
  if numChannels == 1 then
    ret = torch.expand(image, 3, height, width):clone()
  else
    ret = image:clone()
  end

  if color == 'red' then
    ret[{ 1, {}, {1, 2} }] = 1.0
    ret[{ {2, 3}, {}, {1, 2} }] = 0.0
    ret[{ 1, {}, {width-1, width} }] = 1.0
    ret[{ {2, 3}, {}, {width-1, width} }] = 0.0
    ret[{ 1, {1, 2}, {} }] = 1.0
    ret[{ {2, 3}, {1, 2}, {} }] = 0.0
    ret[{ 1, {height-1, height}, {} }] = 1.0
    ret[{ {2, 3}, {height-1, height}, {} }] = 0.0
  elseif color == 'green' then
    ret[{ 2, {}, {1, 2} }] = 1.0
    ret[{ 1, {}, {1, 2} }] = 0.0
    ret[{ 3, {}, {1, 2} }] = 0.0
    ret[{ 2, {}, {width-1, width} }] = 1.0
    ret[{ 1, {}, {width-1, width} }] = 0.0
    ret[{ 3, {}, {width-1, width} }] = 0.0
    ret[{ 2, {1, 2}, {} }] = 1.0
    ret[{ 1, {1, 2}, {} }] = 0.0
    ret[{ 3, {1, 2}, {} }] = 0.0
    ret[{ 2, {height-1, height}, {} }] = 1.0
    ret[{ 1, {height-1, height}, {} }] = 0.0
    ret[{ 3, {height-1, height}, {} }] = 0.0
  end
  return ret
end

function draw(x_seq, firstVidIdx, numPastFrames, numFutureFrames, saveImagesRoot)
  lstm.base:evaluate()
  
  local content_rep, pose_reps = get_reps(x_seq, numPastFrames + numFutureFrames)

  -- generations with predicted pose vectors
  local pose_reps_gen = lstm:fp_pred(pose_reps, content_rep)
  content_rep = nn.utils.addSingletonDimension(content_rep, 3)
  content_rep = nn.utils.addSingletonDimension(content_rep, 4)
  local gens = {}
  for i=1, numFutureFrames do
    local pose_rep = pose_reps_gen[numPastFrames+i]
    pose_rep = nn.utils.addSingletonDimension(pose_rep, 3)
    pose_rep = nn.utils.addSingletonDimension(pose_rep, 4)
    local gen 
    if opt.unet then
      gen = netEC:forward({x_content, pose_rep})[1]
    else
      gen = netD:forward({content_rep, pose_rep})
    end
    table.insert(gens, gen:clone())
  end

  -- generations with ground truth pose vectors
  local gens_gt = {}
  for i=1, numFutureFrames do
    local pose_rep = pose_reps[numPastFrames+i]
    pose_rep = nn.utils.addSingletonDimension(pose_rep, 3)
    pose_rep = nn.utils.addSingletonDimension(pose_rep, 4)

    local gen_gt 
    if opt.unet then
      gen = netEC:forward({x_content, pose_rep})[1]
    else
      gen = netD:forward({content_rep, pose_rep})
    end
    table.insert(gens_gt, gen:clone())
  end

  -- plot frame generated from predicted pose
  local N = math.min(opt.batchSize, 10)
  for i=1, N do
    local pred_frames = {}
    for j=1, numPastFrames do
      table.insert(pred_frames, x_seq[j][i])
    end
    for j=1, numFutureFrames do
      table.insert(pred_frames, gens[j][i])
    end

    -- plot ground truth sequence
    local gt_frames = {}
    for j=1, numPastFrames do
      table.insert(gt_frames, x_seq[j][i])
    end
    for j=1, numFutureFrames do
      table.insert(gt_frames, x_seq[numPastFrames+j][i])
    end

    videoId = i + firstVidIdx - 2
    os.execute(('mkdir -p %s/%04d'):format(saveImagesRoot, videoId))
    for k=1, numPastFrames+numFutureFrames do
      -- Draw actual frames for later evaluation
      image.save(('%s/%04d/raw_pred_%04d.png'):format(saveImagesRoot, videoId, k-1), pred_frames[k])
      image.save(('%s/%04d/raw_gt_%04d.png'):format(saveImagesRoot, videoId, k-1), gt_frames[k])

      local pred_vis
      if k > numPastFrames then
        pred_vis = addBorder(pred_frames[k], 'red')
      else
        pred_vis = addBorder(pred_frames[k], 'green')
      end
      local gt_vis = addBorder(gt_frames[k], 'green')
      image.save(('%s/%04d/pred_%04d.png'):format(saveImagesRoot, videoId, k-1), pred_vis)
      image.save(('%s/%04d/gt_%04d.png'):format(saveImagesRoot, videoId, k-1), gt_vis)
      local both_image = image.toDisplayTensor({gt_vis, pred_vis})
      image.save(('%s/%04d/both_%04d.png'):format(saveImagesRoot, videoId, k-1), both_image)
    end

    -- Write prediction video
    ffmpegCmd = ('ffmpeg -f image2 -framerate 7 -i %s/%04d/pred_%%04d.png %s/%04d/pred.gif -y')
      :format(saveImagesRoot, videoId, saveImagesRoot, videoId)
    rmCmd = ('rm %s/%04d/pred*.png'):format(saveImagesRoot, videoId)
    os.execute(ffmpegCmd)
    os.execute(rmCmd)

    -- Write GT video
    ffmpegCmd = ('ffmpeg -f image2 -framerate 7 -i %s/%04d/gt_%%04d.png %s/%04d/gt.gif -y')
      :format(saveImagesRoot, videoId, saveImagesRoot, videoId)
    rmCmd = ('rm %s/%04d/gt*.png'):format(saveImagesRoot, videoId)
    os.execute(ffmpegCmd)
    os.execute(rmCmd)

    -- Write side-by-side visualization video
    ffmpegCmd = ('ffmpeg -f image2 -framerate 7 -i %s/%04d/both_%%04d.png %s/both_%04d.gif -y')
      :format(saveImagesRoot, videoId, saveImagesRoot, videoId)
    rmCmd = ('rm %s/%04d/both*.png'):format(saveImagesRoot, videoId)
    os.execute(ffmpegCmd)
    os.execute(rmCmd)
  end
end

print('Generating short-term predictions')
local lstmModelPath = (opt.lstmModelFmt):format(opt.modelSliceName, opt.name)
local lstmNets = torch.load(lstmModelPath)
lstmBase = lstmNets['lstm']
lstm = makeLSTM(lstmBase, opt.nPast, opt.nFuture)

for firstVidIdx=1, testLoader.N, opt.batchSize do
  local batch = testLoader:getBatch(opt.batchSize, opt.nPast+opt.nFuture)
  draw(batch, firstVidIdx, opt.nPast, opt.nFuture, saveTestImagesRoot)
end

print('Generating long-term predictions')
lstm = makeLSTM(lstmBase:clone(), opt.nPast, opt.nFutureLong)
for firstVidIdx=1, longLoader.N, opt.batchSize do
  local batch = longLoader:getBatch(opt.batchSize, opt.nPast+opt.nFutureLong)
  draw(batch, firstVidIdx, opt.nPast, opt.nFutureLong, saveLongImagesRoot)
end