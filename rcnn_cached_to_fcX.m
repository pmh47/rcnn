function feat = rcnn_cached_to_fcX(feat, opts, rcnn_model)
% feat = rcnn_pool5_to_fcX(feat, layer, rcnn_model)
%   On-the-fly conversion of pool5 features to fc6 or fc7
%   using the weights and biases stored in rcnn_model.cnn.layers.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% no-op for layer <= 5
if opts.layer > 5
  if strcmp(opts.cached_layer, 'conv5')
    for i = 6:opts.layer
      % weights{1} = matrix of CNN weights [input_dim x output_dim]
      % weights{2} = column vector of biases
      feat = max(0, bsxfun(@plus, feat*rcnn_model.cnn.layers(i).weights{1}, ...
                          rcnn_model.cnn.layers(i).weights{2}'));
	end
  elseif strcmp(opts.cached_layer, 'fc6')
	  if opts.layer ~= 6
		  % ** if opts.layer == 7, we should really compute that!
		  error('opts.cached_layer = fc6, but required layer != 6');
	  end
  elseif strcmp(opts.cached_layer, 'fc7')
	  if opts.layer ~= 7
		  error('opts.cached_layer = fc7, but required layer != 7');
	  end
  else
	  error('opts.cached_layer must be either conv5, fc6 or fc7');
  end
end
