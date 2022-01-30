
f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D52_14ft_run1.sigmf-data",'r');
% f = fopen("/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/Analysis/cores_one.bin");
% f = fopen("5_fake_packets.bin");
rx = fread(f, 'double');
fclose(f);


% rx = [zeros(4000,1); rx]; % Append zeros
% rx = [zeros(4000,1); rx(1:500000)]; % Take subset
% rx = [rx(1:500000)]; % Take subset

i = rx(1:2:end);
q = rx(2:2:end);
rx = complex(i,q);


% Get matlab's detection decisions on a per index basis
offset = 0;
threshold = 1.0;
bandwidth=5e6;
[startOffset,M] = wlanPacketDetect(rx,"CBW20", offset,threshold);
% [startOffset,M] = wlanPacketDetect(rx,"CBW5", offset,threshold);
display(startOffset);
% plot(M)
xlabel('Samples')
ylabel('Decision Statistics')

indices = [];
offset = 1;
max_search=100000; % kinda bogus: matlab will still do an exhaustive search even if only one index requested in find
while 1
    start = find(M(offset:offset+max_search) > 0.99, 1);
    
    if isempty(start)
        disp("No start found");
        break;
    end
    
    start = start(1)+offset-1;
    offset = start;
    finish = find(M(offset:offset+max_search) < 0.5, 1);

    if isempty(finish)
        disp("No finish found");
        break;
    end

    finish = finish(1)+offset-1;
    offset = finish;
    
    indices(end+1) = start;
    display(start);
end
