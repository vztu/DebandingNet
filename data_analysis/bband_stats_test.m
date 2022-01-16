close all;
clear;
warning('off','all');

addpath(genpath('./bband-adaband'))

%% Configs %%
curr_dir = pwd;
data_names = {'2160p'};

for nn = 1:length(data_names)

data_name = data_names{nn};
data_dir = fullfile('../data/test_videos', data_name);
result_dir = 'result';
% video parameters
% if different framerates, should store in array [30,25,25,24..] and modify the
% call of video_framerate in Main Body to video_framerate(i)~
vid_width = 3840;
vid_height = 2160;
% set frame sampling step. 15 means 1/15frames
% sampling_step = 15;
% choose which metric to evaluate: 'BBAD', 'FCDR'
eval_flags = { 'BBAD' };
% eval_flags = { 'BBAD', 'FCDR' };
%%%%%%%%%%%%%

%% Main body
% init
if ~exist(result_dir, 'dir')
   mkdir(result_dir);
end

% parse data names
test_seqs = dir(data_dir);
test_seqs = {test_seqs.name};
% dir results always like  '.','..','video1.mp4','video2.mp4',...
% so we start from 3 to exclude '.','..'
test_seqs = test_seqs(3:end);
times = [];

% video level banding score
banding_score_BBAD_V = zeros(1, length(test_seqs));
% frame level banding score
banding_score_BBAD_I_frames = cell(1, length(test_seqs));

% create temp dir to store decoded videos
video_tmp = 'tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end

save_bband_outputs = false;

fid = fopen([data_name, '_bband_stats.txt'],'w');

for i = 1:length(test_seqs)
    video_name = fullfile(data_dir, test_seqs{i});
    [filepath, name, ext] = fileparts(video_name);
    fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, video_name);

    % extracted frames from videos
    yuv_name = fullfile(video_tmp, ['curr_video', '.yuv']);
    % decode video and store in temp dir
    if ~strcmp(video_name, yuv_name) 
        cmd = ['ffmpeg -loglevel error -y -i ', video_name, ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
        system(cmd);
    end  
    
    % Try to open test_video; if cannot, return
    test_file = fopen(yuv_name,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
            floor(file_length/vid_width/vid_height/1.5));
    % get framerate
    framerate = 60;
    sampling_step = round(framerate / 2);
    % get frame number
    frame_start = 5; 
    frame_end = (floor(file_length/vid_width/vid_height/1.5)-3);
    
    % banding detect data
    banding_score_BBAD_I_frames{1,i} = [];
    im_prev_frame = [];
    t_time = 0;
    % read one frame
    for j = frame_start:sampling_step:frame_end

        % Read frames i-i, i (note that frame_start must be > 0)
        prev_YUV_frame = YUVread(test_file, [vid_width vid_height], j-1);
        this_YUV_frame = YUVread(test_file, [vid_width vid_height], j);

        % color conversion
        im_y = this_YUV_frame(:,:,1);
        % Evaluate BBAD
        if find(ismember(eval_flags, 'BBAD'))
            algo = 'BBAD';
            tic;
            [band_vis_map, band_edge_list, band_score, grad_mag, grad_dir] = ...
                            BBAD_I(im_y, 1);
            t_time = t_time + toc;
            % modulated by temporal score
            if isempty(im_prev_frame)
                ti = 0;
            else
                ti = std2(im_y - prev_YUV_frame(:,:,1));
            end
            % get TI masking weight
            w_ti = exp( - (ti / 20) ^ 2 ) ;
            % calculate BBAD_I score for curr frame
            banding_score_BBAD_I_frames{1,i}(end+1) = w_ti * band_score;
            % write results
            if save_bband_outputs
                save_bband_dir = fullfile(result_dir, 'BBAND_output', name);
                if ~exist(save_bband_dir, 'dir'), mkdir(save_bband_dir); end
                mat_name_out = fullfile(save_bband_dir, sprintf('frame%04d_BBAD.mat',j));
                save(mat_name_out, 'band_vis_map', 'band_edge_list', ...
                     'band_score', 'grad_mag', 'grad_dir');
            end
        end
    end
    fprintf('%f time elapsed...', t_time);
    times(i) = t_time;
    banding_score_BBAD_V(1,i) = mean(banding_score_BBAD_I_frames{1,i});
    fprintf( fid, '%s,%f\n', video_name, banding_score_BBAD_V(1,i));
end
fprintf('\n===\nTotal time: %f; mean time: %f', sum(times), mean(times));
fclose(fid);

end

%% Read one frame from YUV file
function YUV = YUVread(f,dim,frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    Y=fread(f,dim(1)*dim(2),'uchar');
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double');
    
    % Read U-component
    U=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double');
    U=imresize(U,2.0);
    
    % Read V-component
    V=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double');
    V=imresize(V,2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end


%% Plot Stats