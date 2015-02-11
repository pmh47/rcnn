function imdb = imdb_from_sun(sunRootPath, image_set, classNames)

	% Based on rbg's imdb_from_voc.m

	%imdb.name = 'voc_train_2007'
	%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
	%imdb.extension = '.jpg'
	%imdb.image_ids = {'000001', ... }
	%imdb.sizes = [numimages x 2]
	%imdb.classes = {'aeroplane', ... }
	%imdb.num_classes
	%imdb.class_to_id
	%imdb.class_ids
	%imdb.eval_func = pointer to the function that evaluates detections
	%imdb.roidb_func = pointer to the function that returns regions of interest

	cache_file = ['./imdb/cache/imdb_sun_' image_set];
	try
	  load(cache_file);
	catch

	  imdb.name = ['sun_' image_set];
	  imdb.image_dir = [sunRootPath '/JPEGImages'];
	  imdb.annotations_dir = [sunRootPath '/Annotations'];
	  imdb.image_ids = textread([sunRootPath '/ImageSets/Main/' image_set '.txt'], '%s');
	  imdb.extension = 'jpg';
	  imdb.classes = classNames;
	  imdb.num_classes = length(imdb.classes);
	  imdb.class_to_id = ...
		containers.Map(imdb.classes, 1:imdb.num_classes);
	  imdb.class_ids = 1:imdb.num_classes;

	  % VOC specific functions for evaluation and region of interest DB
	  imdb.eval_func = @() error('eval_func not implemented!');
	  imdb.roidb_func = @() error('roidb_func not implemented!');
	  imdb.image_at = @(i) ...
		  sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);

	  for i = 1:length(imdb.image_ids)
		tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
		info = imfinfo([imdb.image_dir '/' imdb.image_ids{i} '.' imdb.extension]);
		imdb.sizes(i, :) = [info.Height info.Width];
	  end

	  fprintf('Saving imdb to cache...');
	  save(cache_file, 'imdb', '-v7.3');
	  fprintf('done\n');
	end
end
