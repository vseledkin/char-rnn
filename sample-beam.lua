
--[[

This file samples characters from a trained model

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model with beam search')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-beam',2,'beam width')
cmd:option('-beamsample',1,'sample from the beam rather than overwriting the lowest probability candidate')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the network using a given sequence, before we sample')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-debug',0,'set to 1 to print extensive beam search state information (very slow)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- debug print: print debug messages if -debug is set
function dprint(str)
    if opt.debug == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU.')
        gprint('If it was trained with CPU use -gpuid -1 for sampling as well.')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab -- e.g. {'a' = 1, 'b' = 2, ...}
local ivocab = {} -- inverse vocab, e.g. {1 = 'a', 2 = 'b', ...}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state = {}
local num_layers = checkpoint.opt.num_layers
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    -- h_init is a 1-D tensor of zeroes, of length = rnn_size, cloned out twice to each layer of the current_state:
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state -- state_size is the number of 1D tensors in current_state.

-- do a few seeded timesteps
local seed_text = opt.primetext
local prediction
if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
		for char_code, c in pairs(UTF8ToCharArray(seed_text)) do
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]]) -- write c
        if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
     -- prediction is a tensor; convert it to cuda type if using GPU:
    if opt.gpuid >= 0 then prediction = prediction:cuda() end
end

-- Print a given node of the string tree to the screen.
-- Strings are stored as a branching linked list of characters.
function printNode(node)
    local currentStringTail = node
    local backwardString = {}
    while currentStringTail do
        backwardString[#backwardString + 1] = currentStringTail.value
        currentStringTail = currentStringTail.parent
    end
    for i = #backwardString, 1, -1 do
        io.write(ivocab[backwardString[i]])
    end
    io.flush()
end

-- Print the portion of the string on which the beam has reached consensus so far.
function printFinalizedCharacters(stringTails)
    if (#stringTails == 1) then
        if (stringTails[1].parent ~= nil) then
            -- Automatically print if there's only one string tail (i.e. opt.beam == 1).
            printNode(stringTails[1].parent)
            stringTails[1].parent = nil
        end
    else
        local tailIterators = {}
        local count = 0
        for k, v in ipairs(stringTails) do
            if (v.parent ~= nil and tailIterators[v.parent] == nil) then
                tailIterators[v.parent] = true
                count = count + 1
            end
        end
        local lastTail;
        -- Trace the string heads backward until they form a common trunk.
        while count > 1 do
            count = 0
            local newTailIterators = {}
            for stringTail, _ in pairs(tailIterators) do
                if (stringTail.parent ~= nil and newTailIterators[stringTail.parent] == nil) then
                    newTailIterators[stringTail.parent] = true
                    count = count + 1
                    lastTail = stringTail.parent
                end
            end
            tailIterators = newTailIterators
        end
        -- Print that trunk, and then chop it off.
        if lastTail ~= nil and lastTail.parent ~= nil then
            printNode(lastTail.parent) -- Print through here.
            lastTail.parent = nil -- Cut the trunk off.
        end
    end
end

-- Function to boost probabilities (multiplicatively and equally) when they're getting too low.
-- Otherwise they eventually all round down to zero.
-- It is important that the probabilities maintain their relative proportions, not that they be correct absolutely.
function boostProbabilities(prob_list)
    local max = 0
    for probIndex,currentProb in ipairs(prob_list) do
        if currentProb > max then
            max = currentProb
        end
    end
    while max < 0.0001 do
        for i = 1,#prob_list do
            prob_list[i] = prob_list[i] * 1000
        end
        max = max * 1000
    end
end

-- start sampling/argmaxing/beam searching
local states = {} -- stores the best opt.beam activation states
local cum_probs = {} -- stores the corresponding cumulative probabilities (periodically boosted with boostProbabilities)
local stringTails = {} -- stores the corresponding string tails generated by the states so far
-- initially populate states table with the net
states[#states + 1] = current_state
current_state = nil
cum_probs[#cum_probs + 1] = 1

local timer = torch.Timer()

for outputIndex=1, opt.length do
    dprint("\nPicking character #" .. outputIndex)
	local newStateIndices = {}
	local newCumProbs = {}
	local newStringTails = {}
	for stateIndex,stateContent in ipairs(states) do
        if (outputIndex > 1 or stateIndex > 1) then -- The state was already loaded above if this is the first character.
            -- Pull the previous character.
            prev_char = torch.Tensor{stringTails[stateIndex].value}
            if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
            -- Forward the latest character and extract the probabilities that result.
            if opt.debug == 1 then print("state #" .. stateIndex .. ", forwarding character '" .. ivocab[prev_char[1]] .. "'") end
            local lst = protos.rnn:forward{prev_char, unpack(stateContent)}
            local newStateContent = {}
            for i=1,state_size do table.insert(newStateContent, lst[i]:clone()) end -- clone to avoid entangling with other entries in state[].
            states[stateIndex] = newStateContent -- Save the modified state back to the state table.
            prediction = lst[#lst] -- log probabilities
        end
        -- Get the probabilities of each character at the current state.
        prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one and are actual probabilities
        -- Populate currentBestChars with the top opt.beam characters, in order of likelihood.
        local currentBestChars = {}
        local probsCopy = probs:clone() -- Clone probabilities tensor so we can zero out items as we draw them.
        for candidate=1, opt.beam do
            local char_index
            if opt.sample == 0 then
                -- Pull the highest-probability character index.
                local _, prev_char = probsCopy:max(1)
                char_index = prev_char[1]
            else
                -- Sample a character index.
                prev_char = torch.multinomial(probsCopy:float(), 1):resize(1):float()
                char_index = prev_char[1]
            end
            if opt.debug == 1 then print("state #" .. stateIndex .. ", option #" .. candidate .. ": "
                        .. char_index .. " ('" .. ivocab[char_index] .. "'); prob: " .. probs[char_index]) end
            probsCopy[char_index] = 0 -- Zero out that index so we don't pull it again.
            currentBestChars[#currentBestChars + 1] = char_index -- Add it to the list of best characters at this node.
        end
        -- For each of the characters in currentBestChars, check its probability and keep a rolling
        -- record of the best states in newStateIndices. How many states to keep is defined by opt.beam.
        for _, char_index in ipairs(currentBestChars) do
            local cumProb = probs[char_index] * cum_probs[stateIndex] -- Cumulative probability of this character choice
            if cumProb > 0 then -- If cumProb is equal to zero, this is a dead end.
                local insertionPoint = -1
                if #newStateIndices < opt.beam then
                     -- If newStateIndices has fewer entries than the beam width, we automatically qualify.
                    insertionPoint = #newStateIndices + 1
                else
                    local probsTensor = torch.Tensor(newCumProbs);
                    if opt.beamsample == 0 then
                        -- Find the lowest cumulative probability and its index in newCumProbs.
                        local _, min = probsTensor:min(1)
                        local minIndex = min[1]
                        if (probsTensor[minIndex] <= cumProb) then
                            insertionPoint = minIndex
                        end
                    else
                        -- Sample the beam states and randomly draw a low one.
                        probsTensor:div(opt.temperature) -- scale by temperature
                        probsTensor:div(torch.sum(probsTensor)) -- renormalize so probs sum to one
                        -- Since we want a low one, we first have to invert the probabilities in the tensor.
                        local min = probsTensor:min(1); local minValue = min[1]
                        local max = probsTensor:max(1); local maxValue = max[1]
                        probsTensor = -probsTensor + maxValue + minValue

                        local min = probsTensor:min(1); local max = probsTensor:max(1)
                        if (max[1] <= 0 or min[1] < 0) then
                            print("Error with probabilities tensor: \n", probsTensor)
                        end
                        local index_ = torch.multinomial(probsTensor, 1):resize(1):float()
                        local index = index_[1]
                        if (newCumProbs[index] <= cumProb) then
                            insertionPoint = index
                        end
                    end
                end
                if insertionPoint > 0 then
                    newStateIndices[insertionPoint] = stateIndex;
                    newCumProbs[insertionPoint] = cumProb
                    local newStringTail = {parent = stringTails[stateIndex], value = char_index}
                    newStringTails[insertionPoint] = newStringTail
                end
            end
        end
	end

    -- Replace the old states with the new.
    local newStates = {}
    for iterator, newIndex in ipairs(newStateIndices) do
        dprint("Entry " .. iterator .. ": Cloning from state index ".. newIndex)
        newStates[iterator] = states[newIndex]
    end

    states = newStates;
    cum_probs = newCumProbs;
    stringTails = newStringTails;
    if (opt.debug == 1) then
        for stateIndex=1,#stringTails do
            dprint(string.format("          State #%i, prob %.3e:", stateIndex, cum_probs[stateIndex]))
            printNode(stringTails[stateIndex])
            io.write('\n'); io.flush()
        end
    end
    -- Boost the probabilities if they're getting too low; all that matters is relative probabilities,
    -- and with long enough text they could otherwise exceed floating point accuracy and zero out completely.
    boostProbabilities(cum_probs)
    -- Print however many characters the beam has reached consensus about,
    -- but do it less frequently with a wider beam to avoid churning needlessly.
    if outputIndex % opt.beam == 0 then printFinalizedCharacters(stringTails) end
    -- Periodically take out the trash.
    if outputIndex % 10 == 0 then collectgarbage() end
end

-- Pick the winning state.
local max = 0
local winningIndex = 0
for probIndex,currentProb in ipairs(cum_probs) do
    if currentProb > max then
        max = currentProb
        winningIndex = probIndex
    end
end
-- Transcribe the winning string.
printNode(stringTails[winningIndex])

local time = timer:time().real
if (opt.verbose == 1) then
    print(string.format("\nSample completed in %.2f seconds.", time))
end
