disp('Graph Results of Tracking');

%need to swap x and y and add 1 to each axis 
%matlab reads the axis a little bit wierd

track_json = './data/tracks_frame.json';
track_result = jsondecode(fileread(track_json));

data = './data/raw_data/Segmentation_and_result/%d/%d_3Dconnection2.fig';
data_save = './data/raw_data/Segmentation_and_result/%d/%d_tracked.fig';


%iterate through alll frames
fields = fieldnames(track_result);
for k=1:(numel(fields)-1)
      fprintf('\nOn Frame: %d', k);
      %open the correct figure
      f = openfig(sprintf(data, k, k));
      delete(findall(gcf,'type','text'))
      
      %iterate through all tracks in frame
      frame = track_result.(fields{k + 1});
      for i=1:numel(frame)
          track = frame(i);
          %need to swap x and y and add 1 to each axis 
          %matlab is index by 1
          centroid = track.centroid + 1;
          text('String',num2str(track.id),'Position',[centroid(2) centroid(1) centroid(3)]);
      end
      savefig(f, sprintf(data_save, k, k));
      delete(f);
end

