% path="/mnt/wd500GB/CSC500/csc500-main/datasets/KRI-16Devices-RawData/38ft/WiFi_air_X310_3123D70_38ft_run1.sigmf-data";
% f = fopen(path,'r');
% rx = fread(f, 'double');


% path = "/media/steven/main/blessed_dataset_2/bin/day-2_transmitter-18_transmission-7.bin";
% path = "/media/steven/main/blessed_dataset_2/bin/day-2_transmitter-19_transmission-7.bin";
% path = "/media/steven/main/blessed_dataset_2/bin/day-3_transmitter-14_transmission-1.bin";
% path = "/media/steven/main/blessed_dataset_2/bin/day-3_transmitter-19_transmission-6.bin";
% path = "/media/steven/main/blessed_dataset_2/bin/day-3_transmitter-3_transmission-1.bin";
% path = "/media/steven/main/blessed_dataset_2/bin/day-8_transmitter-5_transmission-6.bin";

path = "/mnt/wd500GB/CSC500/csc500-main/csc500-dataset-preprocessor/Analysis/metehan.bin";


f = fopen(path,'r');
rx = fread(f, 'double');

fclose(f);


% rx = [zeros(4000,1); rx]; % Append zeros
% rx = [zeros(4000,1); rx(1:500000)]; % Take subset
% rx = [rx(1:500000)]; % Take subset

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

fs = 5e6;
% fs = 25e6;


offset = 0;
% threshold = 1-10*eps;
threshold = 0.90;

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
