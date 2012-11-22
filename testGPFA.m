% Test case for three latent factors plus known stimulus terms
% AE 2012-10-19

% create toy example
rng(1)
% [grd, Y, S] = GPFA.toyExample();
% [grd, Y, S] = GPFA.toyExampleOri('gauss');
[grd, Y, S] = GPFA.toyExampleOri('poisson');
grd = grd.normFactors();

% fit model
model = GPFA('Tolerance', 1e-6);
model = model.fit(Y, S, grd.p);
model = model.normFactors();


%% diagnostic plots
cc = model.C' * grd.C;
[~, ndx] = max(abs(cc));
for i = 1 : model.p
    subplot(3, model.p, i)
    cla, hold on
    plot(grd.X(i, :), 'k')
    plot(sign(cc(ndx(i), i)) * model.X(ndx(i), :), 'r')
    axis tight
    plot(repmat((1 : model.N - 1) * model.T, 2, 1), ylim, 'k')
    ylabel(sprintf('Factor %d', i))
    xlabel('Time')
    xlim([0 100])
end

for i = 1 : model.p
    subplot(3, model.p, model.p + i)
    cla, hold on
    plot(grd.C(:, i), 'k')
    plot(sign(cc(ndx(i), i)) * model.C(:, ndx(i)), 'r')
    ylabel(sprintf('Loading %d', i))
    xlabel('Neuron #')
    axis tight
end

legend({'Ground truth', 'Model fit'})

subplot(3, model.p, 2 * model.p + 1)
cla, hold on
plot(grd.D(:), model.D(:), '.k')
axis tight
xlabel('Stim term: ground truth')
ylabel('Stim term: model fit')
shg
