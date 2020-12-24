% The directories for a specific protein track
video_file ='./data/test/4064/4064_trackVid.avi';
data_tracked = './data/test/4064/%d_tracked.fig';
events_json = './data/test/4064/4064results.json';

% TODO put this in a single for loop
% -------------------------------------------------------------------------
% Create the zoomed frames using the camera settings from first frame 
% Expects a track directory with the first frame containing the correct 
% camera settings and applises those settings to all frames, and saves
% them.
% -------------------------------------------------------------------------
% 
% display('Modify the first frame to get the desired view');
% display('!!!!!!!!!!!!!!!SAVE IT!!!!!!!!!!!!!!!');
% display('Press Enter when done');
% 
% openfig(sprintf(data_tracked, 1),'visible');
% pause;
% 
% f=openfig(sprintf(data_tracked, 1),'invisible');
% childHandle = get(f, "Children");
% cPos = get(childHandle(2), 'CameraPosition');
% cTarg = get(childHandle(2), 'CameraTarget');
% cVector = get(childHandle(2), 'CameraUpVector');
% xLim = get(childHandle(2), 'XLim');
% yLim = get(childHandle(2), 'YLim');
% cLim = get(childHandle(2), 'CLim');
% 
% 
% for fNum=2:70
%     
%     fprintf('\nOn Frame: %d', fNum);
%     % open the figure f
%     g=openfig(sprintf(data_tracked, fNum),'invisible');
%     g_child = get(g, "Children");
%     %change camera to previous view only need to change these params
%     set(g_child(2), 'CameraPosition', cPos);
%     set(g_child(2), 'CameraTarget', cTarg);
%     set(g_child(2), 'CameraUpVector', cVector);
%     set(g_child(2), 'XLim', xLim);
%     set(g_child(2), 'YLim', yLim);
%     set(g_child(2), 'CLim', cLim);
%     
%     savefig(sprintf(data_tracked, fNum));
%     %delete the new figure
%     delete(g);
% end



% -------------------------------------------------------------------------
% This uses the zoomed frames to create a video. Uses the track directory
% with the zoomed frames for the specific protein track.
% -------------------------------------------------------------------------
events = jsondecode(fileread(events_json));
fields = fieldnames(events);

v = VideoWriter(video_file);
% number of frames per second
v.FrameRate = 0.5;
open(v);

width = 0;
height = 0;

for fNum=1:70
    % print the progress 
    fprintf('\nOn Frame: %d', fNum);
    % open the figure f
    f=openfig(sprintf(data_tracked, fNum),'invisible');

    % add a titile containing the frame num
    title(sprintf('Frame: %d  Event: %s', fNum, events.(fields{fNum})));
    
    if fNum==1
        % find the framesize using the first frame
        frame = getframe(gcf);
        width = size(frame.cdata, 1);
        height = size(frame.cdata, 2);
    end

    
    % dont include null frames
    if isempty(events.(fields{fNum}))
        delete(f);
        continue;
    end
    
    % set the position and size for getframe otherwise each figure might be
    % different
    set(gcf, 'Position',  [100 100 height width])
    frame = getframe(gcf);
    writeVideo(v, frame);
    %delete the new figure
    delete(f);
    
    if strcmp(events.(fields{fNum}), 'DEAD')
        break
    end
    
end

close(v)

