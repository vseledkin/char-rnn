require "./Print.lua"
local LSTMEX = {}
function LSTMEX.lstm(input_size, rnn_size, n, dropout,mems)
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
      input_size_L = rnn_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CMulTable()({i2h, h2h})
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

    local i2hm = nn.Linear(input_size_L, 4 * mems)(x):annotate{name='i2h_'..L}
    local h2hm = nn.Linear(rnn_size, 4 * mems)(prev_h):annotate{name='h2h_'..L}
    local all_input_sumsm = nn.CMulTable()({i2hm, h2hm})
    local reshapedm = nn.Reshape(4, mems)(all_input_sumsm)
    local n1m, n2m, n3m, n4m = nn.SplitTable(2)(reshapedm):split(4)

    -- decode the gates

    n1 = nn.Replicate(1,2)(n1)
    n1m = nn.Replicate(1,3)(n1m)

    n2 = nn.Replicate(1,2)(n2)
    n2m = nn.Replicate(1,3)(n2m)

    n3 = nn.Replicate(1,2)(n3)
    n3m = nn.Replicate(1,3)(n3m)

    n4 = nn.Replicate(1,2)(n4)
    n4m = nn.Replicate(1,3)(n4m)

    local in_gate = nn.Sigmoid()(nn.MM()({n1m,n1}))
    --in_gate = nn.PrintSize("in_gate")(in_gate)
    local forget_gate = nn.Sigmoid()(nn.MM()({n2m,n2}))
    --forget_gate = nn.PrintSize("forget_gate")(forget_gate)
    local out_gate = nn.Sigmoid()(nn.MM()({n3m,n3}))
    --out_gate = nn.PrintSize("out_gate")(out_gate)
    -- decode the write inputs
    local in_transform = nn.Tanh()(nn.MM()({n4m,n4}))
    --in_transform = nn.PrintSize("in_transform")(in_transform)
    -- perform the LSTM update
    -- select prev mem block


    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    next_c = nn.PrintTensor(100,"Memory")(next_c)
    --next_c = nn.PrintSize("next_c")(next_c)


    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
--[[
    local i2hr = nn.Linear(input_size_L, rnn_size)(x):annotate{name='i2h_'..L}
    local h2hr = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h_'..L}
    local read_controller = nn.CAddTable()({i2hr, h2hr})

    local attn = nn.MV()({next_h, read_controller}) -- batch_l x source_l x 1
    attn = nn.SoftMax()(attn):annotate{name="Attention"}
    next_h = nn.MV(true)({next_h,attn})
--]]
    next_h = nn.SpatialAveragePooling(2, 2)(next_h)
    next_h = nn.PrintSize("next_h")(next_h)
    next_h = nn.Reshape(-1,mems*rnn_size/4)(next_h)
    next_h = nn.Squeeze()(next_h)
    --next_h = nn.PrintSize("next_h")(next_h)
    next_h = nn.Linear(mems*rnn_size/4,rnn_size)(next_h)
    --next_h = nn.PrintSize("next_h")(next_h)
    --next_h = nn.Print("next_h")(next_h)
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
