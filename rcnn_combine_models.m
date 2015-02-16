function rcnn_combine_models(imdb, model_names, final_model_suffix)
	
	conf = rcnn_config('sub_dir', imdb.name);
	assert(iscell(model_names) && length(model_names) > 1);
	
	rcnn_model = rcnn_load_model([conf.cache_dir model_names{1}], conf.use_gpu);
	
	function result = merge_cols(left, right, right_indices)
		result = left;
		result(:, right_indices) = right;
	end
	
	for model_index = 2 : length(model_names)
		
		next_model = rcnn_load_model([conf.cache_dir model_names{model_index}], conf.use_gpu);
		
		rcnn_model.detectors.W = merge_cols(rcnn_model.detectors.W, next_model.detectors.W, next_model.class_ids);
		rcnn_model.detectors.B = merge_cols(rcnn_model.detectors.B, next_model.detectors.B, next_model.class_ids);
		
		rcnn_model.SVs.keys_neg = merge_cols(rcnn_model.SVs.keys_neg, next_model.SVs.keys_neg, next_model.class_ids);
		rcnn_model.SVs.scores_neg = merge_cols(rcnn_model.SVs.scores_neg, next_model.SVs.scores_neg, next_model.class_ids);
		
		rcnn_model.classes = [rcnn_model.classes ; next_model.classes];
		rcnn_model.class_ids = [rcnn_model.class_ids , next_model.class_ids];
		
	end
	
	if ~isempty(final_model_suffix) && final_model_suffix(1) ~= '_'
		final_model_suffix = ['_' final_model_suffix];
	end
	
	save([conf.cache_dir 'rcnn_model' final_model_suffix], 'rcnn_model');
	
end

