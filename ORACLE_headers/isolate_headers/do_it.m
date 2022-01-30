rootdir = '/mnt/wd500GB/CSC500/csc500-super-repo/datasets/KRI-16Devices-RawData';
filelist = dir(fullfile(rootdir, '**/*.sigmf-data'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list


M = containers.Map('KeyType','char', 'ValueType','any');
for i = 1:length(filelist)
% for i = 1:2
    path = strcat(filelist(i).folder, '/', filelist(i).name);
    

    indices = get_80211a_indices(path);

    M(path) = indices;
    disp(i/length(filelist));
end

j = jsonencode(M);
f = fopen("indices.json",'w');
rx = fwrite(f, j);
fclose(f);
display("Done");

% keySet = {'Li','Jones','Sanchez'};
% testLi = [5.8 7.35];
% testJones = [27 3.92 6.4 8.21];
% testSanchez = 'C:\Tests\Sanchez.dat';
% 
% valueSet = {testLi,testJones,testSanchez};
% M = containers.Map(keySet,valueSet,'UniformValues',false);