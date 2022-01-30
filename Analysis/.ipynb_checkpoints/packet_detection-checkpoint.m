f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D52_14ft_run1.sigmf-data",'r');
% f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/Analysis/cores_one.bin");
whole_file = fread(f, 'double');
fclose(f);

rx = whole_file(1:5000);

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

fs = 5e6;
% fs = 25e6;


offset = 0;
% threshold = 1-10*eps;
threshold = 0.95;

bandwidth=5e6;

% 'CBW5' – Channel bandwidth of 5 MHz
% 'CBW10' – Channel bandwidth of 10 MHz
% 'CBW20' – Channel bandwidth of 20 MHz
% 'CBW40' – Channel bandwidth of 40 MHz
% 'CBW80' – Channel bandwidth of 80 MHz
% 'CBW160' – Channel bandwidth of 160 MHz
% 'CBW320' – Channel bandwidth of 320 MHz

[startOffset,M] = wlanPacketDetect(rx,"CBW20", offset,threshold);
display(startOffset);
plot(M)
xlabel('Samples')
ylabel('Decision Statistics')

return;
