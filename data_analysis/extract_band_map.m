close all;
clear;
warning('off','all');

%% Configs %%
curr_dir = pwd;
data_dir = '../datasets/VP9BandingDataset/train/input';
result_dir = '../datasets/VP9BandingDataset/train/input_bandmap';
lib_dir = './';
% video parameters
% if different framerates, should store in array [30,25,25,24..] and modify the
% call of video_framerate in Main Body to video_framerate(i)~
width = 1920;
height = 1080;
%%%%%%%%%%%%%

% init
if ~exist(result_dir, 'dir')
   mkdir(result_dir);
end

addpath(genpath(lib_dir))

% parse data names
test_seqs = dir(data_dir);
test_seqs = {test_seqs.name};
% dir results always like  '.','..','video1.mp4','video2.mp4',...
% so we start from 3 to exclude '.','..'
test_seqs = test_seqs(3:end);
times = [];

for i = 1:length(test_seqs)
    video_name = fullfile(data_dir, test_seqs{i});
    [filepath, name, ext] = fileparts(video_name);
    fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, video_name);
    img_rgb = imread(video_name);
    im_yuv = reshape(convertRgbToYuv(reshape(img_rgb, width * height, 3)), height, width, 3);
    tic;
    [band_vis_map, band_edge_list, band_score, grad_mag, grad_dir] = ...
                    BBAD_I(im_yuv(:,:,1), 1);
    toc;
    band_vis_map = band_vis_map ./ max(max(band_vis_map));
    band_map_name = fullfile(result_dir, [name, '.png']);
    imwrite(band_vis_map, band_map_name);
end
    
    
function val = clipValue(val, valMin, valMax)
% check if value is valid

for i = 1 : 1 : size(val(:))
	if val(i) < valMin
		val(i) = valMin;
	elseif val(i) > valMax
		val(i) = valMax;
	end
end
end

function yuv = convertRgbToYuv(rgb)
% convert row vector RGB [0, 255] to row vector YUV [0, 255]

rgbToYuv =  [0.2989    0.5866    0.1145;
   -0.1688   -0.3312    0.5000;
    0.5000   -0.4184   -0.0816];

rgb = double(rgb);

yuv = (rgbToYuv * rgb.').';
yuv(:, 2 : 3) = yuv(:, 2 : 3) + 127;

yuv = uint8(clipValue(yuv, 0, 255));
end

