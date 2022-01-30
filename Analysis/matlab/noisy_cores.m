f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/Analysis/cores_one.bin");
rx = fread(f, 'double');
fclose(f);

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

awgn_power_dBW = 1e-1000;
before_awgn = wgn(256,1,awgn_power_dBW,'dBm', 'complex');
after_awgn = wgn(256,1,awgn_power_dBW,'dBm', 'complex');

before_awgn = 0.001 * before_awgn;
after_awgn = 0.001 * after_awgn;

rx = [before_awgn; rx; after_awgn];

fs = 25e6;

[p,f,t] = pspectrum(rx,fs,'spectrogram');

waterfall(f,t,p');
ax = gca;
ax.XDir = 'reverse';
view(30,45);

offset = 0;
threshold = 1;
[startOffset,M] = wlanPacketDetect(rx,"CBW5", offset,threshold);
plot(M)
xlabel('Samples')
ylabel('Decision Statistics')