function [instance_frequencies, image_frequencies] = rcnn_get_gt_class_frequencies(imdb, varargin)

% Each of the return values is indexed by elements of imdb.class_ids; instance_frequencies
% is the number of gt boxes in that class, while image_frequencies is the number of images
% containing at least one gt box of the class
	
ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('cache_name', 'v1_finetune_voc_2007_trainval_iter_70000', @isstr);
ip.addParamValue('cached_layer', 'pool5', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;
inst
% X_pos and keys_pos are indexed by elements of imdb.class_ids, so may in principal
% be 'sparse' if imdb.class_ids is non-contiguous
[~, keys_pos] = rcnn_get_positive_gt_features(imdb, opts.cache_name, opts.cached_layer);

maximum_class_id = max(imdb.class_ids);
instance_frequencies = zeros(maximum_class_id, 1);
image_frequencies = zeros(maximum_class_id, 1);

for class_id = imdb.class_ids
	
	instance_frequencies(class_id) = size(keys_pos{class_id}, 1);
	if instance_frequencies(class_id) > 0
		image_frequencies(class_id) = size(unique(keys_pos{class_id}(:, 1)), 1);
	else
		image_frequencies(class_id) = 0;
	end
	
end
