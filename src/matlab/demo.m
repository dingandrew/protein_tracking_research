function [] = demo()
% Plot different plots according to slider location.
% open initial figure
S.fh = openfig('./data/labled_frames/1_tracked.fig'); 
set(gcf,'name','FRAME: 1','numbertitle','off')
% Create textbox
S.ann = annotation(gcf,'textbox',...
                [0.005 0.10 0.1 0.8],...
                'String',{'Enter ID to see Info about protein cluster','Probably should MAXMIZE WINDOW'},...
                'FitBoxToText','off');

S.sl = uicontrol('style','slide',...
                 'unit','pix',...
                 'position',[20 10 500 30],...
                 'min',1,'max',70,'val',1,...
                 'sliderstep',[1/69 1/10],...
                 'callback',{@sl_call,S});  
             

slbl = uicontrol(gcf,'Style','text', 'string', 'Protein ID:', 'FontSize', 13);
slbl.Position = [555 10 100 30];

S.el = uicontrol('Style','edit', 'callback',{@el_call,S});
S.el.Position = [655 10 100 30];


global track_result
track_json = './data/tracks_pretty.json';
track_result = jsondecode(fileread(track_json));


savefig('./data/demo.fig');
  
function [] = el_call(varargin)
% Callback for the edit text0.
global track_result
[edit, h, S] = varargin{[1, 2,3]};  % calling handle and data structure.

id = str2double(get(edit, 'String'));
fprintf('Get Track Info for ID: %d\n', id)

if isnan(id)
   set(S.ann,'String','Please enter an valid Integer ID');
else
   field_name = sprintf('x%d_0', id);
   indx = numel(track_result.(field_name));
%    display(indx);
%    display(track_result.(field_name))
   info = track_result.(field_name)(indx);
   set(S.ann,'String',{sprintf('Track ID:%d', id),...
                       newline, ...
                       sprintf('Last Frame:%d', info.Frame),...
                       newline, ...
                       sprintf('Num Points:%d', info.locs),...
                       newline, ...
                       strcat('Centroid: ', num2str(info.centroid)),...
                       newline, ...
                       strcat('State:', info.state),...
                       newline, ...
                       strcat('Origin:', info.origin),...
                       newline, ...
                       'If still active on last frame, this means that the protein did not die but merged or split.'});
end
    
drawnow; %force graphics update


             
function [] = sl_call(varargin)
% Callback for the slider.
[h,S] = varargin{[1,3]};  % calling handle and data structure.
cla
data_tracked = './data/labled_frames/%d_tracked.fig';

fNum = round(get(h,'value'));
fprintf('ON FRAME: %d\n', fNum);
if fNum > 0 && fNum < 71
    %open new figure
    set(gcf,'name',sprintf('FRAME: %d', fNum),'numbertitle','off')
    g = openfig(sprintf(data_tracked, fNum),'invisible');
    %copy the new figure objects to old figure
    copyobj(g.Children, S.fh);
    %display(g.Children)
    %delete the new figure
    delete(g);
%     display(S.fh.Children);
    %display(get(S.fh.Children(6), 'CameraPosition'))

    
    %if true that means the view has changed delete contextmenu
    if numel(S.fh.Children) == 8     
        delete(S.fh.Children(1)); 
    end
    
   
    %change camera to previous view only need to change these params
    set(S.fh.Children(5), 'CameraPosition', get(S.fh.Children(7), 'CameraPosition'));
    set(S.fh.Children(5), 'CameraTarget', get(S.fh.Children(7), 'CameraTarget'));
    set(S.fh.Children(5), 'CameraUpVector', get(S.fh.Children(7), 'CameraUpVector'));
    set(S.fh.Children(5), 'XLim', S.fh.Children(7).XLim);
    set(S.fh.Children(5), 'YLim', S.fh.Children(7).YLim);
    set(S.fh.Children(5), 'CLim', S.fh.Children(7).CLim);

    delete(S.fh.Children(7));

else
    disp('cannot plot');
end

