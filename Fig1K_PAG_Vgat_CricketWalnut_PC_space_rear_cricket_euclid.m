%PC space for cricket and walnut -- do eating / approaching / walking
%cluster?

clear all

%walnut folders
   folders{1,1} = 'E:\PAG_VGAT_hunt\8_21_2020\635';
   folders{2,1} = 'E:\PAG_VGAT_hunt\8_21_2020\636';
   folders{3,1} = 'E:\PAG_VGAT_hunt\8_21_2020\637';
   folders{4,1} = 'E:\PAG_VGAT_hunt\8_21_2020\641';    

%cricket folders   
   folders{1,2} = 'E:\PAG_VGAT_hunt\8_24_2020\635';
   folders{2,2} = 'E:\PAG_VGAT_hunt\8_24_2020\636';
   folders{3,2} = 'E:\PAG_VGAT_hunt\8_24_2020\637';
   folders{4,2} = 'E:\PAG_VGAT_hunt\8_24_2020\641';  
   
%coreg folders (walnut first column, cricket second column
   coreg_folders{1} = 'E:\PAG_VGAT_hunt\coreg_cricketArtPrey_walnut\635\2_0';
   coreg_folders{2} = 'E:\PAG_VGAT_hunt\coreg_cricketArtPrey_walnut\636\2_0';
   coreg_folders{3} = 'E:\PAG_VGAT_hunt\coreg_cricketArtPrey_walnut\637\2_0';
   coreg_folders{4} = 'E:\PAG_VGAT_hunt\coreg_cricketArtPrey_walnut\641\2_0';  

%%

for mouseNum = 1:size(folders,1)

    cd(folders{mouseNum,1})
    load('output_CNMF-E.mat','neuron')
    sig = neuron.C_raw;
    sig_temp{1} = sig';
    load('eating_vars.mat','detect_indice','eating_indice')
    approachIndicesMS_temp{1} = detect_indice; eatIndicesMS_temp{1} = eating_indice; clearvars detect_indice eating_indice
    load('BehaviorMS_Rear.mat','rearingIndicesMS')    
    rearingIndicesMS_temp{1} = rearingIndicesMS; clearvars rearingIndicesMS;
            
    cd(folders{mouseNum,2})
    load('output_CNMF-E.mat','neuron')    
    sig = neuron.C_raw;
    sig = sig';
    load('BehaviorMS_2.mat','approachIndicesMS','eatIndicesMS')
    load('BehaviorMS_Rear.mat','rearingIndicesMS')

            load('Tracking.mat'); mouseVel = Tracking.mouseVelMS;
            load('fracSessArtPrey.mat')
            sessLength = length(neuron.C_raw);
            OF_Indices = 1:round(.15 .* sessLength); %first 15% of session is open field.
            Cricket_Indices = (round(.15 .* sessLength))+1:round(fracSessArtPrey.*sessLength);            
            sig = sig(Cricket_Indices(1):Cricket_Indices(end),:); %remove artificial prey data.
            mouseVel = mouseVel(Cricket_Indices(1):Cricket_Indices(end));
            approachIndicesMS = approachIndicesMS(Cricket_Indices(1):Cricket_Indices(end));
            eatIndicesMS = eatIndicesMS(Cricket_Indices(1):Cricket_Indices(end)); 
            rearingIndicesMS = rearingIndicesMS(Cricket_Indices(1):Cricket_Indices(end));            
            walkingIndices = mouseVel > 4;
            walkingIndices(find(approachIndicesMS)) = 0; %dont use approach indices
    approachIndicesMS_temp{2} = approachIndicesMS; eatIndicesMS_temp{2} = eatIndicesMS; rearingIndicesMS_temp{2} = rearingIndicesMS;
    sig_temp{2} = sig;
    clearvars sig approachIndicesMS rearingIndicesMS eatIndicesMS
    
    approachIndicesMS = [approachIndicesMS_temp{1};approachIndicesMS_temp{2}];
    eatIndicesMS = [eatIndicesMS_temp{1};eatIndicesMS_temp{2}];
    rearingIndicesMS = [rearingIndicesMS_temp{1};rearingIndicesMS_temp{2}];
    
    cd(coreg_folders{mouseNum})
    load('cellRegistered.mat','cell_registered_struct')
    coreg = cell_registered_struct.cell_to_index_map;
    idxToDel = find(coreg(:,1)==0 | coreg(:,2)==0); coreg(idxToDel,:) = [];
    sig_temp{1} = sig_temp{1}(:,coreg(:,1));
    sig_temp{2} = sig_temp{2}(:,coreg(:,2));
    sig = [sig_temp{1};sig_temp{2}];
    
            
% De-mean
sig = bsxfun(@minus,sig,mean(sig));
% Do the PCA
[coeff,score,latent,tsquared,explained,mu] = pca(sig);

figure(32)
subplot(1,4,mouseNum)
%plot ONLY EATING TIMEPOINTS
plot3(score(find(eatIndicesMS),1),score(find(eatIndicesMS),2),score(find(eatIndicesMS),3),'.','Color','g'); hold on;

%plot ONLY APPROACHING TIMEPOINTS
plot3(score(find(approachIndicesMS),1),score(find(approachIndicesMS),2),score(find(approachIndicesMS),3),'.','Color','r')

%plot ONLY REARING TIMEPOINTS
plot3(score(find(rearingIndicesMS),1),score(find(rearingIndicesMS),2),score(find(rearingIndicesMS),3),'.','Color','b')

title(['CRICKET/WALNUT coreged; app:red, eat:green, rear:blue'])        
 
%CALCULTE THE SILHOUETTE SCORE FOR THE CLUSTERS:
clusterID = nan(length(score),1);
clusterID(find(approachIndicesMS)) = 1;
clusterID(find(eatIndicesMS)) = 2;
clusterID(find(rearingIndicesMS)) = 3;
idxToDel = find(isnan(clusterID));

clusterID(idxToDel) = [];

X = score(:,1:3);
X(idxToDel,:) = [];

s = silhouette(X,clusterID);
silhouetteScore(mouseNum) = nanmean(s);

%calculate a null distribution of silhouette scores
iter=1000;
for iterNum = 1:iter
    clusterShuff = clusterID(randperm(length(clusterID)));
    s_nullDist(iterNum) = nanmean(silhouette(X,clusterShuff));
end

figure(34)
subplot(size(folders,1),1,mouseNum)
hist(s_nullDist,30); hold on;
plot([silhouetteScore(mouseNum) silhouetteScore(mouseNum)],[0 30],'Color','r')
ylabel('iteration count')
xlabel('sihouette score')
box off
xlim([-.2 .6])


% calculate the distance of each point from its cluster's centroid

clust{1} = score(find(eatIndicesMS),[1:3]);
clust{2} = score(find(approachIndicesMS),[1:3]);
clust{3} = score(find(rearingIndicesMS),[1:3]);

for clustNum = 1:length(clust)
   temp = clust{clustNum};
   centroid = nanmean(temp); 
   
   for sampleNum = 1:length(temp)
        distCentroid(sampleNum) = pdist([temp(sampleNum,:);centroid]);
   end
   if ~exist('distCentroid')
      distCentroid = []; 
   end
   
   if clustNum==1
       distCentroid_Eat{mouseNum} = distCentroid; clearvars distCentroid
   elseif clustNum==2
       distCentroid_App{mouseNum} = distCentroid; clearvars distCentroid
   elseif clustNum==3
       distCentroid_Rear{mouseNum} = distCentroid; clearvars distCentroid
   end 
end

end

%% Calculate mean dist from centroid for each behavior

%first combine across mice
distCentroid_App_All = [];
distCentroid_Eat_All = [];
distCentroid_Rear_All = [];

for mouseNum = 1:length(folders)
    distCentroid_App_All = [distCentroid_App_All,distCentroid_App{mouseNum}];
    distCentroid_Eat_All = [distCentroid_Eat_All,distCentroid_Eat{mouseNum}];
    distCentroid_Rear_All = [distCentroid_Rear_All,distCentroid_Rear{mouseNum}]; 
end

meanAll = [nanmean(distCentroid_App_All),nanmean(distCentroid_Eat_All),nanmean(distCentroid_Rear_All)];
seAll = [nanstd(distCentroid_App_All)./sqrt(length(distCentroid_App_All)),nanstd(distCentroid_Eat_All)./sqrt(length(distCentroid_Eat_All)),nanstd(distCentroid_Rear_All)./sqrt(length(distCentroid_Rear_All))];

figure(390)
labels = {'approach','eat','rear'};
bar(meanAll); hold on;
errorbar(meanAll,seAll,'LineStyle','none','Color','k')
set(gca, 'XTickLabel', labels); box off;
ylabel('mean Euclid. dist. betw. samples and behavior centroid (top 3 PCs)')
[p,h,stats] = ranksum(distCentroid_App_All,distCentroid_Eat_All)
text(1.5,30,['p=' num2str(round(p,4))],'Color','r')

title('walnut / cricket concatenated')

%%
figure(57)
meanS = nanmean(silhouetteScore);
seS = nanstd(silhouetteScore) ./ sqrt(size(folders,1));
bar(meanS); hold on;
errorbar(meanS,seS,'LineStyle','none','Color','k'); hold on;
scatter(ones(1,size(folders,1)),silhouetteScore,10,'filled')
ylim([0 .6])
ylabel('silhouette scores')