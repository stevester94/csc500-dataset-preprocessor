
function indices = get_80211a_indices(path)
%     path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D52_14ft_run1.sigmf-data"
    display(strcat("Processing: ", path));
%     indices = [1,20000];
%     return;
    f = fopen(path,'r');
    rx = fread(f, 'double');
    fclose(f);
    
    i = rx(1:2:end);
    q = rx(2:2:end);
    rx = complex(i,q);
    
    
    % Get matlab's detection decisions on a per index basis
    offset = 0;
    threshold = 1.0;
    [startOffset,M] = wlanPacketDetect(rx,"CBW20", offset,threshold);
    
    indices = uint64([]);
    offset = 1;
    max_search=100000; % kinda bogus: matlab will still do an exhaustive search even if only one index requested in find
    while 1
        end_find = min([offset+max_search, length(M)]);
        start = find(M(offset:end_find) > 0.99, 1);
        
        if isempty(start)
            break;
        end
        
        start = start(1)+offset-1;
        offset = start;
        end_find = min([offset+max_search, length(M)]);
        finish = find(M(offset:end_find) < 0.5, 1);
    
        if isempty(finish)
            break;
        end
    
        finish = finish(1)+offset-1;
        offset = finish;
        
        indices(end+1) = uint64(start);
    end
end

