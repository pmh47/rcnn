function [X_pos, keys_pos] = rcnn_get_positive_gt_features(imdb, cache_name, cached_layer)

% Factored out of rcnn_train. Features are returned for the same layer as stored in the cache

save_file = sprintf('./feat_cache/%s/%s/gt_pos_%s_cache.mat', ...
	cache_name, imdb.name, cached_layer);
try
	load(save_file);
	fprintf('Loaded saved positives from ground truth boxes\n');
catch
	[X_pos, keys_pos] = get_positive_features(imdb, cache_name, cached_layer);
	save(save_file, 'X_pos', 'keys_pos', '-v7.3');
end
	
function [X_pos, keys] = get_positive_features(imdb, cache_name, cached_layer)
	
% Note that this includes all classes, not just those requested for training, so
% if a cache needs to be written, it is always complete

X_pos = cell(max(imdb.class_ids), 1);
keys = cell(max(imdb.class_ids), 1);

for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));

  d = rcnn_load_cached_features(cache_name, cached_layer, ...
      imdb.name, imdb.image_ids{i});

  for j = imdb.class_ids
    if isempty(X_pos{j})
      X_pos{j} = single([]);
      keys{j} = [];
    end
    sel = find(d.class == j);
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, d.feat(sel,:));
      keys{j} = cat(1, keys{j}, [i*ones(length(sel),1) sel]);
    end
  end
end
