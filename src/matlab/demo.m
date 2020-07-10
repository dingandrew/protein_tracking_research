function [] = demo()
% Plot different plots according to slider location.

S.fh = openfig('./data/raw_data/Segmentation_and_result/1/1_3Dconnection2.fig'); 

S.sl = uicontrol('style','slide',...
                 'unit','pix',...
                 'position',[20 10 500 30],...
                 'min',1,'max',70,'val',1,...
                 'sliderstep',[1/10 1/10],...
                 'callback',{@sl_call,S});  
             
             
function [] = sl_call(varargin)
% Callback for the slider.
[h,S] = varargin{[1,3]};  % calling handle and data structure.
cla
switch round(get(h,'value'))
      case 1
          %open new figure
          g = openfig('./data/raw_data/Segmentation_and_result/1/1_3Dconnection2.fig'...
                       ,'invisible');
          %copy the new figure objects to old figure
          copyobj(g.Children, S.fh);
          %delete the new figure
          delete(g);
          %need to delete old color bar and axes at end of list
          %axes is last element delting that automatically deletest its
          %colorbar
          delete(S.fh.Children(5));

      case 2
          cla
          close(S.fh)
          S.fh = openfig('./data/raw_data/Segmentation_and_result/2/2_3Dconnection2.fig');          
      case 3
          cla
          close(S.fh)
          S.fh = openfig('./data/raw_data/Segmentation_and_result/3/3_3Dconnection2.fig');            
      otherwise
          disp('cannot plot');
end