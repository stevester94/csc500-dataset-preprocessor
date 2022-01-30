import wlan.*
f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D52_14ft_run1.sigmf-data",'r');
% f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/Analysis/cores_one.bin");
whole_file = fread(f, 'double');
fclose(f);

rx = whole_file(1:50000);

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

fs = 5e6;
chanBW = 'CBW20';
num_samps = 50000;


% Create a WaveformAnalyzer object to parse and analyze the packet within a waveform
analyzer = WaveformAnalyzer;
process(analyzer,rx,chanBW,fs);

% Display the summary of the detected packets
detectionSummary(analyzer);

macSummary(analyzer,1);