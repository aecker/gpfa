function startup()
% Add folders that are needed to MATLAB path
% AE 2012-10-29

folder = fileparts(mfilename('fullpath'));
addpath(folder)
addpath(fullfile(folder, 'lib'));
addpath(fullfile(folder, 'lib/invToeplitz'));
