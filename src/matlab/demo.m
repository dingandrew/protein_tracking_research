function [] = demo()
% Plot different plots according to slider location.
% open initial figure
S.fh = openfig('./data/labled_frames/1_tracked.fig'); 
set(gcf,'name','FRAME: 1','numbertitle','off')
S.sl = uicontrol('style','slide',...
                 'unit','pix',...
                 'position',[20 10 500 30],...
                 'min',1,'max',70,'val',1,...
                 'sliderstep',[1/69 1/10],...
                 'callback',{@sl_call,S});  
savefig('./data/demo.fig');
             
             
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
%     display(g.Children)
    %delete the new figure
    delete(g);
%     display(S.fh.Children);
%     display(get(S.fh.Children(6), 'CameraPosition'))

    
    %if true that means the view has changed delete contextmenu
    if numel(S.fh.Children) == 6     
        delete(S.fh.Children(1)); 
    end
    
    %change camera to previous view only need to change these params
    set(S.fh.Children(3), 'CameraPosition', get(S.fh.Children(5), 'CameraPosition'));
    set(S.fh.Children(3), 'CameraTarget', get(S.fh.Children(5), 'CameraTarget'));
    set(S.fh.Children(3), 'CameraUpVector', get(S.fh.Children(5), 'CameraUpVector'));
%     set(S.fh.Children(3), 'CameraViewAngle', get(S.fh.Children(5), 'CameraViewAngle'));
%     set(S.fh.Children(3), 'View', S.fh.Children(5).View);
%     set(S.fh.Children(3), 'DataAspectRatio', S.fh.Children(5).DataAspectRatio);
%     set(S.fh.Children(3), 'PlotBoxAspectRatio', S.fh.Children(5).PlotBoxAspectRatio);
    set(S.fh.Children(3), 'XLim', S.fh.Children(5).XLim);
    set(S.fh.Children(3), 'YLim', S.fh.Children(5).YLim);
    set(S.fh.Children(3), 'CLim', S.fh.Children(5).CLim);
%     set(S.fh.Children(3), 'XTick', S.fh.Children(5).XTick);
%     set(S.fh.Children(3), 'XTickLabel', S.fh.Children(5).XTickLabel);
%     set(S.fh.Children(3), 'YTick', S.fh.Children(5).YTick);
%     set(S.fh.Children(3), 'YTickLabel', S.fh.Children(5).YTickLabel);
%     set(S.fh.Children(3), 'ZTick', S.fh.Children(5).ZTick);
%     set(S.fh.Children(3), 'ZTickLabel', S.fh.Children(5).ZTickLabel); 
%     set(S.fh.Children(3), 'Position', S.fh.Children(5).Position);
%     set(S.fh.Children(3), 'InnerPosition', S.fh.Children(5).InnerPosition);
%     set(S.fh.Children(3), 'OuterPosition', S.fh.Children(5).OuterPosition);
%     set(S.fh.Children(3), 'SortMethod', S.fh.Children(5).SortMethod);
%     set(S.fh.Children(3), 'TickDir', S.fh.Children(5).TickDir);  
    
%     need to delete old color bar and axes at end of list
%     axes is last element deleting that automatically delete its
%     colorbar
    delete(S.fh.Children(5));

else
    disp('cannot plot');
end

