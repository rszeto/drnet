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
  --learningRate     (default 0.002)             learning rate  
  --beta1            (default 0.9)               momentum term for adam
  -b,--batchSize     (default 100)               batch size  
  -g,--gpu           (default 0)                 gpu to use  
  --name             (default 'default')         checkpoint name
  --dataRoot         (default 'data/new_mnist')  data root directory
  --optimizer        (default 'adam')            optimizer to train with
  --nEpochs          (default 300)               max training epochs  
  --seed             (default 1)                 random seed  
  --epochSize        (default 10000)             number of samples per epoch  
  --imageSize        (default 64)                size of image
  --dataset          (default new_moving_mnist)               dataset
  --movingDigits     (default 1)
  --cropSize         (default 227)               size of crop (for kitti only)
  --normalize                                    if set normalize predicted pose vectors to have unit norm
  --rnnSize          (default 256)
  --rnnLayers        (default 2)
  --nThreads         (default 5)                 number of dataloading threads
  --dataPool         (default 200)
  --dataWarmup       (default 10)
  --nPast            (default 10)                number of frames to condition on.  
  --nFuture          (default 5)                 number of frames to predict.
  --printEvery       (default 100)               Print stats after this many batches
  --plotEvery        (default 1000)              Plot images after this many batches
  --testEvery        (default 50)                Evaluate after this many batches
  --sliceName        (string)                    Name of the new MNIST slice to train on
  --modelRootFmt     (default 'logs/new_moving_mnist/%s')
  --trainOnPreds     (default true)              Whether to propagate predicted poses during training
  --testOnPreds      (default true)              Whether to propagate predicted poses during testing
  --patience         (default 20)                Number of epochs to wait for before early stopping
]]

opt.modelRoot = (opt.modelRootFmt):format(opt.sliceName)
opt.save = opt.modelRoot .. '/lstm/' .. opt.name
os.execute('mkdir -p ' .. opt.save .. '/gen/')

progressLogPath = ('%s/progress.log'):format(opt.save)
os.execute('rm ' .. progressLogPath)
progressLog = io.open(progressLogPath, 'a')
progressLog:write('epoch#\tbatch#\ttrain_mse\tval_mse\n')

assert(optim[opt.optimizer] ~= nil, 'unknown optimizer: ' .. opt.optimizer)
opt.optimizer = optim[opt.optimizer]

-- setup some stuff
torch.setnumthreads(1)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.gpu + 1)
print('<gpu> using device ' .. opt.gpu)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
math.randomseed(opt.seed)

opt.modelPath = opt.modelRoot .. '/model.t7'
local nets = torch.load(opt.modelPath)

opt.nShare = nets.opt.nShare
opt.contentDim = nets.opt.contentDim
opt.poseDim = nets.opt.poseDim
opt.T = opt.nPast + opt.nFuture + 1
opt.batchSize = nets.opt.batchSize
opt.geometry = nets.opt.geometry
opt.imageSize = nets.opt.imageSize 
opt.movingDigits = nets.opt.movingDigits


require 'data.data'

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
write_opt(opt)

netEC:training()
netEP:training()

require 'models.lstm'
lstm = makeLSTM()

local criterion = {}
for i=1,opt.nPast+opt.nFuture do
  criterion[i] = nn.MSECriterion()
  criterion[i]:cuda()
end

local x_content = {}
for i=1, opt.nShare do
  x_content[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

local x = {}
for i=1, opt.nPast + opt.nFuture + 1 do
  x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

local optimState = {learningRate = opt.learningRate, beta=opt.beta1}

function squeeze_all(input)
  for i=1, #input do
    input[i] = torch.squeeze(input[i])
  end
  return input
end

function get_reps(x_seq)
  for i=1, opt.nShare do
    x_content[i]:copy(x_seq[i])
  end

  for i=1,opt.nPast + opt.nFuture + 1 do
    x[i]:copy(x_seq[i]) 
  end

  local pose_reps = {}
  for i=1,opt.nPast+opt.nFuture+1 do 
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


function plot(x_seq, fname, epoch, iter)
  lstm.base:evaluate()
  
  local content_rep, pose_reps = get_reps(x_seq)

  -- generations with predicted pose vectors
  local pose_reps_gen = lstm:fp_pred(pose_reps, content_rep)
  content_rep = nn.utils.addSingletonDimension(content_rep, 3)
  content_rep = nn.utils.addSingletonDimension(content_rep, 4)
  local gens = {}
  for i=1, opt.nFuture do
    local pose_rep = pose_reps_gen[opt.nPast+i]
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
  for i=1, opt.nFuture do
    local pose_rep = pose_reps[opt.nPast+i]
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
  local to_plot = {}
  local N = math.min(opt.batchSize, 5)
  for i=1, N do
    for j=1, opt.nPast do
      table.insert(to_plot, x_seq[j][i])
    end
    for j=1, opt.nFuture do
      table.insert(to_plot, gens[j][i])
    end

    -- -- plot frame generated from GT pose
    -- for j=1, opt.nPast do
    --   table.insert(to_plot, x_seq[j][i])
    -- end
    -- for j=1, opt.nFuture do
    --   table.insert(to_plot, gens_gt[j][i])
    -- end

    -- plot ground truth sequence
    for j=1, opt.nPast do
      table.insert(to_plot, x_seq[j][i])
    end
    for j=1, opt.nFuture do
      table.insert(to_plot, x_seq[opt.nPast+j][i])
    end
  end

  borderPlot(to_plot)
  local img = image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=opt.nPast + opt.nFuture}
  image.save(('%s/gen/%s_epoch-%02d_iter-%02d.png'):format(opt.save, fname, epoch, iter), img)
end

function train(x_seq)
  lstm.base:training()

  lstm.grads:zero()

  local content_rep, pose_reps = get_reps(x_seq)

  local gen_pose, in_pose
  if opt.trainOnPreds then
    gen_pose, in_pose = lstm:fp_pred(pose_reps, content_rep)
  else
    gen_pose, in_pose = lstm:fp_obs(pose_reps, content_rep)
  end
  local dgen_pose = {}
  local err = 0
  for t=1,opt.nPast+opt.nFuture do
    err = err + criterion[t]:forward(gen_pose[t], pose_reps[t+1])
    dgen_pose[t] = criterion[t]:backward(gen_pose[t], pose_reps[t+1])
  end
  lstm:bp(pose_reps, content_rep, dgen_pose)

  table.insert(train_err, err/(opt.nPast+opt.nFuture))

  opt.optimizer(function() return 0, lstm.grads end, lstm.params, optimState)
end

function test(x_seq)
  local content_rep, pose_reps = get_reps(x_seq)

  local gen_pose
  if opt.testOnPreds then
    gen_pose = lstm:fp_pred(pose_reps, content_rep)
  else
    gen_pose = lstm:fp_obs(pose_reps, content_rep)
  end
  local err = 0
  for t=1,opt.nPast+opt.nFuture do
    err = err + criterion[t]:forward(gen_pose[t], pose_reps[t+1])
  end

  table.insert(test_err, err/(opt.nPast+opt.nFuture))
end

local val_batch = valLoader:getBatch(opt.batchSize, opt.T)
local bestTestErr = 1e10
local meanTrainErr, meanTestErr

for epoch=0, opt.nEpochs do
  train_err = {}
  test_err = {}

  local batch_num = 0
  for iter=1, opt.epochSize, opt.batchSize do
    local batch = trainLoader:getBatch(opt.batchSize, opt.T)
    train(batch)

    if batch_num % opt.testEvery == 0 then 
      local batch = valLoader:getBatch(opt.batchSize, opt.T)
      test(batch)
      meanTestErr = torch.Tensor(test_err):mean()
    end

    if batch_num % opt.printEvery == 0 then
      meanTrainErr = torch.Tensor(train_err):mean()
      print(('Epoch: %02d Batch %02d - Speed = %.2f secs/batch, Train Error = %.5f, Test Error = %.5f'):
    	  format(epoch, batch_num, 0, meanTrainErr, meanTestErr))
      progressLog:write(('%04d\t%04d\t%.05f\t%.05f\n'):format(epoch, batch_num, meanTrainErr, meanTestErr))
      progressLog:flush()
      train_err = {}
      test_err = {}
    end

    if batch_num % opt.plotEvery == 0 then
      plot(batch, 'train', epoch, batch_num)
    end

    batch_num = batch_num + 1
  end

  plot(val_batch, 'valid', epoch, 0)
  torch.save(('%s/model.t7'):format(opt.save), {lstm=lstm.base, opt=opt, epoch=epoch, bestTestErr=bestTestErr})

  if meanTestErr < bestTestErr then
    print(('Saving best model with validation MSE %.05f'):format(meanTestErr))
    torch.save(('%s/model_best.t7'):format(opt.save), {lstm=lstm.base, opt=opt, epoch=epoch, bestTestErr=bestTestErr})
    bestTestErr = meanTestErr
    numEpochsSinceImprovement = 0
  else
    numEpochsSinceImprovement = numEpochsSinceImprovement + 1
    if numEpochsSinceImprovement >= opt.patience then
      print(('Validation MSE has not improved for %d iterations, quitting'):format(opt.patience))
      break
    end
  end

end

progressLog:close()