%PC space for cricket and walnut -- do eating / approaching / walking

clear all; close all;

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

assayNum = 2;

minNumBehavs = 1; 
%%

for mouseNum = 1:size(folders,1)

cd(folders{mouseNum,assayNum})
load('output_CNMF-E.mat','neuron')
load('good_neurons.mat')

sig = neuron.C_raw(find(good_neurons),:);
sig = sig';

        if assayNum == 1
            %load('Seg.mat','behavSegAll')
            load('eating_vars.mat','detect_indice','eating_indice','detect_frames','eating_frames')
            approachFrameMS = detect_frames; eatFrameMS = eating_frames; approachIndicesMS = detect_indice; eatIndicesMS = eating_indice; clearvars detect_indice eating_indice
            load('BehaviorMS_Rear.mat','rearingIndicesMS','rearingFrameMS')
        elseif assayNum == 2
            %load('Seg_ManuallyChecked.mat','behavSegAll')
            load('BehaviorMS_2.mat','approachIndicesMS','eatIndicesMS','approachFrameMS','eatFrameMS')
            load('BehaviorMS_Rear.mat','rearingIndicesMS','rearingFrameMS')
        end
        
        %remove behavioral instances if too few behaviors (for Euclidean
        %distance calculation:
        eatIndicesMS_eu = eatIndicesMS;
        approachIndicesMS_eu = approachIndicesMS;
        rearingIndicesMS_eu = rearingIndicesMS;
        
        if size(eatFrameMS,1) < minNumBehavs
            eatIndicesMS_eu = zeros(length(eatIndicesMS),1);
        end
        if size(approachFrameMS,1) < minNumBehavs
            approachIndicesMS_eu = zeros(length(approachIndicesMS),1);
        end
        if size(rearingFrameMS,1) < minNumBehavs
            rearingIndicesMS_eu = zeros(length(rearingIndicesMS),1);
        end

        if assayNum==2
            load('Tracking.mat'); mouseVel = Tracking.mouseVelMS;
            load('fracSessArtPrey.mat')
            sessLength = length(neuron.C_raw);
            OF_Indices = 1:round(.15 .* sessLength); %first 15% of session is open field.
            Cricket_Indices = (round(.15 .* sessLength))+1:round(fracSessArtPrey.*sessLength);
            
            sig = sig(Cricket_Indices(1):Cricket_Indices(end),:); %remove artificial prey data.
            mouseVel = mouseVel(Cricket_Indices(1):Cricket_Indices(end));
            approachIndicesMS = approachIndicesMS(Cricket_Indices(1):Cricket_Indices(end));
            approachIndicesMS_eu = approachIndicesMS_eu(Cricket_Indices(1):Cricket_Indices(end));
            
            eatIndicesMS = eatIndicesMS(Cricket_Indices(1):Cricket_Indices(end)); 
            eatIndicesMS_eu = eatIndicesMS_eu(Cricket_Indices(1):Cricket_Indices(end)); 

            rearingIndicesMS = rearingIndicesMS(Cricket_Indices(1):Cricket_Indices(end));            
            rearingIndicesMS_eu = rearingIndicesMS_eu(Cricket_Indices(1):Cricket_Indices(end));            

            walkingIndices = mouseVel > 4;
            walkingIndices(find(approachIndicesMS)) = 0; %dont use approach indices
        end

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


    if assayNum == 2
        title(['app crick:red, eat crick:green, rear:blue'])        
    elseif assayNum == 1
        title(['approach walnut: red, eat walnut:green'])
    end
 
%CALCULATE THE SILHOUETTE SCORE FOR THE CLUSTERS:
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

% Calculate the distance of each point from its cluster's centroid

clust{1} = score(find(eatIndicesMS_eu),[1:3]);
clust{2} = score(find(approachIndicesMS_eu),[1:3]);
clust{3} = score(find(rearingIndicesMS_eu),[1:3]);

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
[p,h] = ranksum(distCentroid_App_All,distCentroid_Eat_All)
text(1.5,30,['p=' num2str(round(p,4))],'Color','r')

if assayNum==2
    title('cricket')
elseif assayNum==1
    title('walnut')
end

%%
figure(57)
meanS = nanmean(silhouetteScore);
seS = nanstd(silhouetteScore) ./ sqrt(size(folders,1));
bar(meanS); hold on;
errorbar(meanS,seS,'LineStyle','none','Color','k'); hold on;
scatter(ones(1,size(folders,1)),silhouetteScore,10,'filled')
ylim([0 .6])
ylabel('silhouette scores')
[h,p,~,stats] = ttest(silhouetteScore)
tstatVal = stats.tstat;
title(['p=' num2str(round(p,3)) ' tstat=' num2str(tstatVal)],'Color','r')