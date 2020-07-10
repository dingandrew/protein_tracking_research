function [] = demo()
% Plot different plots according to slider location.
% open initial figure
S.fh = openfig('./data/raw_data/Segmentation_and_result/1/1_3Dconnection2.fig'); 

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
data_tracked = './data/raw_data/Segmentation_and_result/%d/%d_tracked.fig';

fNum = round(get(h,'value'));
fprintf('ON FRAME: %d\n', fNum);
if fNum > 0 && fNum < 71
    %open new figure
    g = openfig(sprintf(data_tracked, fNum, fNum),'invisible', 'reuse');
    %copy the new figure objects to old figure
    copyobj(g.Children, S.fh);
%     display(g.Children)
    %delete the new figure
    delete(g);
%     display(S.fh.Children)
%     display(get(S.fh.Children(6), 'CameraPosition'))
    %need to delete old color bar and axes at end of list
    %axes is last element deleting that automatically deletest its
    %colorbar
    
    %if true that means the view has changed keep the previos view by
    %deleting new axis
    if numel(S.fh.Children) == 6     
        delete(S.fh.Children(1));        
        delete(S.fh.Children(5));
    else
        delete(S.fh.Children(5));
    end
%     display(S.fh.Children);

else
    disp('cannot plot');
end

