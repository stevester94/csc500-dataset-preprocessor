f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D52_14ft_run1.sigmf-data",'r');
% f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/Analysis/cores_one.bin");
rx = fread(f, 'double');
fclose(f);

rx = rx(1:256);

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

fs = 5e6;
% fs = 25e6;

% instbw(rx, 5e6);

[p,f,t] = pspectrum(rx,fs,'spectrogram');

waterfall(f,t,p');
ax = gca;
ax.XDir = 'reverse';
view(30,45);
return;