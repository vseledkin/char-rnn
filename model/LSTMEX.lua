require "./Print.lua"
local LSTMEX = {}

-- retunrs memory_dim vector - convex combination of memory slots
function LSTMEX.ReadHead(name, control, memory)
  --control = nn.PrintSize("read-control")(control)
  --memory = nn.PrintSize("memory")(memory)
  local attn = nn.MV()({memory, control}) -- batch_l x source_l x 1
  attn = nn.SoftMax()(attn):annotate{name="Attention"}
  --attn = nn.PrintSize("Attention")(attn)
  --local attn = nn.Max(2)(memory)
    -- make glimpse
   local glimpse = nn.MV(true)({memory,attn}) -- batch_l x 1 x rnn_size
   --glimpse = nn.PrintSize("glimpse")(glimpse)
   --glimpse = nn.Print("glimpse")(glimpse)
  return glimpse
end


-- writes to whole memory weighted decomposition of x ruled by y signal
function LSTMEX.WriteHead(name, control, x, memory, rnn_size, memory_slots)
  --control = nn.PrintSize("write-control")(control)
  --control = nn.Print("write-control")(control)
  --memory = nn.PrintSize("memory")(memory)
--  memory = nn.Print("memory")(memory)
  --local attn = nn.MV()({memory, control}) -- batch_l x source_l x 1
  --attn = nn.CAddTable()({nn.Linear(rnn_size,4)(control),attn})
  local attn = nn.Linear(rnn_size, memory_slots)(control)
  attn = nn.SoftMax()(attn):annotate{name="Attention"}
  --attn = nn.PrintSize("RAttention")(attn)
  --attn = nn.Print("WriteAttention")(attn)
  attn = nn.Replicate(1,3)(attn)
  --attn = nn.PrintSize("REPAttention")(attn)
  --attn = nn.Print("WriteAttention")(attn)
  -- make glimpse
  --local glimpse = nn.MV(true)({memory,attn}) -- batch_l x 1 x rnn_size

  --local tx = nn.Reshape(1,memory_dim,true)(x)
  --local tx = nn.PrintSize('x')(x)
  local tx = nn.Replicate(1,2)(x)
  --tx = nn.PrintSize('tx')(tx)
  --tx = nn.Print('tx')(tx)
  local delta = nn.MM(){attn,tx}
  --delta = nn.PrintSize('delta')(delta)
  --delta = nn.Print('delta')(delta)
  local updated_memory = nn.CAddTable()({delta, memory})
  --updated_memory = nn.PrintSize('updated_memory')(updated_memory)
  --updated_memory = nn.Print('updated_memory')(updated_memory)
  return updated_memory
end

-- Gated eraser by control signal
function LSTMEX.EraseHead(name, control, memory,rnn_size,memory_slots)
  --control = nn.PrintSize("erase-control")(control)
  --memory = nn.PrintSize("memory")(memory)
  --local attn = nn.MV()({memory, control}) -- batch_l x source_l x 1
  local attn = nn.Linear(rnn_size, memory_slots)(control)
--  attn = nn.Linear(4+rnn_size,4)(nn.JoinTable(2)({attn,x}))
  attn = nn.Sigmoid()(attn):annotate{name="Attention"}
  --attn = nn.PrintSize("EAttention")(attn)


  --address = nn.Print('erase_address')(address)
  address = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(attn))
  address = nn.Replicate(rnn_size,3)(address)
  --address = nn.PrintSize('mul_mask')(address)

  --address = nn.PrintSize('replicated_mask')(address)
  local erased_memory = nn.CMulTable()({address, memory})
  --erased_memory = nn.PrintSize('erased_memory')(erased_memory)
  return erased_memory
end

function LSTMEX.lstm(input_size, rnn_size, n, dropout, memory_slots)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = nn.LookupTable(input_size,rnn_size)(inputs[1])
      --x = OneHot(input_size)(inputs[1])
    --x = nn.Print(x)
      input_size_L = rnn_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Tanh()(n1)
    local forget_gate = nn.Tanh()(n2)
    local out_gate = nn.Tanh()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update

    -- erase controlled by forget gate
    local erased_c = LSTMEX.EraseHead('Erase',forget_gate,prev_c,rnn_size,memory_slots)

    -- write controlled by input gate
    local next_c = LSTMEX.WriteHead('Write',in_gate,in_transform,erased_c,rnn_size,memory_slots)
    next_c = nn.PrintTensor(100,"Memory")(next_c)
    --local next_c           = nn.CAddTable()({
    --    nn.CMulTable()({forget_gate, prev_c}),
    --    nn.CMulTable()({in_gate,     in_transform})
    --  })
    --next_c = nn.Print()(next_c)
    -- read controlled by output gate
    local next_h = LSTMEX.ReadHead('Read', out_gate, next_c)
    next_h = nn.Tanh()(next_h)
    --local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTMEX
