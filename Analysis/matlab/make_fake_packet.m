cfgNonHT = wlanNonHTConfig;
txWaveform = wlanWaveformGenerator([1;0;0;1],cfgNonHT, ...
    'NumPackets',5,'IdleTime',20e-6);
rxWaveform = [zeros(4000,1);txWaveform];

interleaved = zeros(length(rxWaveform)*2, 1);

% I dont know matlab
j = 1;
for i = 1:length(rxWaveform)
    interleaved(j) = real(rxWaveform(i));
    j = j + 1;
    interleaved(j) = imag(rxWaveform(i));
    j = j + 1;
end

f = fopen("5_fake_packets.bin",'w');
fwrite(f, interleaved, 'double');
fclose(f);

f = fopen("5_fake_packets.bin",'r');
whole_file = fread(f, 'double');
fclose(f);

% Parse it into complex
rx = whole_file;
i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);

offset = 0;
threshold = 1.0;
[startOffset,M] = wlanPacketDetect(rx,"CBW20", offset,threshold);
display(startOffset);
plot(M)
xlabel('Samples')
ylabel('Decision Statistics')