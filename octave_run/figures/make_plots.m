graphics_toolkit('gnuplot');
D = '/home/user/matlab/octave_run/';
up = load([D 'upper_HN.txt']);
x = up(1,:); Y = up(2:end,:);
lbl = {'f`(\eta)  (nanofluid velocity)', 'F`(\eta)  (dust velocity)', ...
       't(\eta)  (nanofluid temperature)', 'tp(\eta)  (dust temperature)'};
row = [2 5 6 8];
fn  = {'fig1_fp','fig2_Fp','fig3_t','fig4_tp'};
col = {'r','b',[0.8 0.6 0],'k'};
for k=1:4
  h = figure('visible','off');
  plot(x, Y(row(k),:), 'color', col{k}, 'linewidth', 2);
  xlabel('\eta'); ylabel(lbl{k}); grid on;
  title(sprintf('%s   (\\lambda=-1, s=2, \\sigma_1=\\sigma_2=0.2, Gr=0.001, X=0)', lbl{k}));
  print(h, [D 'figures/' fn{k} '.png'], '-dpng', '-r120');
  close(h);
end
disp('plots written');
