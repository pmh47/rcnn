function rcnn_export_scores(rcnn_model, imdb, cacheSuffix, outputPath)

	if isfield(rcnn_model, 'folds')
		error('model must be trained with a single fold only');
	end

	% allBoxes here is a cell array, {classIndex}{imageIndex} -- note indices 
	% rather than actual ids! Rcnn_test strips out groundtruth boxes.
	% Each cell is an array of rows of form [box score], where box is 
	% x1 y1 x2 y2 and score is a scalar detector score for the cell's class.
	% classScoreThresholds is indexed by (classIndex), and indicates the score
	% below which rcnn_test would strip detections (before NMS) for that class.
	% rcnn_test guarantees that for all classes, the boxes will appear in the 
	% same order (and hence should actually be the same)
	[~, allBoxes, classScoreThresholds] = rcnn_test(rcnn_model, imdb, cacheSuffix);
	
	image_ids = imdb.image_ids;
	assert(length(allBoxes) > 0);
	assert(length(image_ids) == length(allBoxes{1}));
	
	for i = 1 : length(image_ids)
		
		fprintf('processing image %d / %d\n', i, length(image_ids));

		% Get the box coordinates for this image, in order [x1 x2 y1 y2]
		imageBoxes = allBoxes{1}{i}(:, [1 3 2 4]);

		% Get the corresponding scores, with one row per box and one column per class
		rawScores = zeros(size(imageBoxes, 1), length(allBoxes));
		for classIndex = 1 : length(allBoxes)
			rawScores(:, classIndex) = allBoxes{classIndex}{i}(:, 5);
		end
		
		% Load the groundtruth class overlaps of each box, as part of the cached features
		d = rcnn_load_cached_features(rcnn_model.training_opts.cache_name, rcnn_model.training_opts.cached_layer, imdb.name, image_ids{i});
		
		% Get the post-threshold and post-NMS scores for each box -- i.e. set to -Inf all 
		% scores that rcnn_test would drop at the fixed-count thresholding or dynamic 
		% NMS steps
		postThresholdScores = rawScores;
		postNmsScores = rawScores;
		for classIndex = 1 : length(allBoxes)
			rawScoresAreBelowThreshold = rawScores(:, classIndex) < classScoreThresholds(classIndex);
			rawScoreIndicesAboveThreshold = find(~rawScoresAreBelowThreshold);
			postThresholdScores(rawScoresAreBelowThreshold, classIndex) = -Inf;
			postThresholdBoxes = allBoxes{classIndex}{i}(rawScoreIndicesAboveThreshold, :);
			postThresholdBoxIndicesToKeepAfterNms = nms(postThresholdBoxes);
			rawBoxIndicesToKeepAfterNms = rawScoreIndicesAboveThreshold(postThresholdBoxIndicesToKeepAfterNms);
			rawBoxIndicesToDropAfterNms = 1 : size(allBoxes{classIndex}{i}, 1);
			rawBoxIndicesToDropAfterNms(rawBoxIndicesToKeepAfterNms) = [];
			postNmsScores(rawBoxIndicesToDropAfterNms, classIndex) = -Inf;
		end		

		% Write the results to disc -- each row has four box coordinates, K scores, K overlaps, 
		% K thresholded scores, and K NMS'ed scores
		csvwrite( ...
			[outputPath '/' image_ids{i}], ...
			[imageBoxes rawScores d.overlaps postThresholdScores postNmsScores] ...
		);

	end

end

