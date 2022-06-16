%PC space for cricket and walnut -- do eating / approaching / walking

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
    
    eatIndicesMS_All{mouseNum} = eatIndicesMS;
    approachIndicesMS_All{mouseNum} = approachIndicesMS;
    
    cd(coreg_folders{mouseNum})
    load('cellRegistered.mat','cell_registered_struct')
    coreg = cell_registered_struct.cell_to_index_map;
    idxToDel = find(coreg(:,1)==0 | coreg(:,2)==0); coreg(idxToDel,:) = [];
    sig_temp{1} = sig_temp{1}(:,coreg(:,1));
    sig_temp{2} = sig_temp{2}(:,coreg(:,2));
    sig = [sig_temp{1};sig_temp{2}];
    
    
for cellNum = 1:size(sig,2)
   sig_z(:,cellNum) = zscore(sig(:,cellNum)); 
end

sig_z_All{mouseNum} = sig_z;
            
% De-mean
sig = bsxfun(@minus,sig,mean(sig));
% Do the PCA
[coeff,score,latent,tsquared,explained,mu] = pca(sig);

figure(32)
subplot(1,4,mouseNum)
%plot ONLY EATING TIMEPOINTS
idx1 = length(rearingIndicesMS_temp{1});
idx2 = length(rearingIndicesMS_temp{2}) + length(rearingIndicesMS_temp{1});

idx1_All(mouseNum) = idx1;
idx2_All(mouseNum) = idx2;

plot3(score(find(eatIndicesMS(1:idx1)),1),score(find(eatIndicesMS(1:idx1)),2),score(find(eatIndicesMS(1:idx1)),3),'.','Color',[0 1 0]); hold on;
plot3(score(find(eatIndicesMS(idx1+1:idx2)),1),score(find(eatIndicesMS(idx1+1:idx2)),2),score(find(eatIndicesMS(idx1+1:idx2)),3),'.','Color',[0 .7 .3]); hold on;

plot3(score(find(approachIndicesMS(1:idx1)),1),score(find(approachIndicesMS(1:idx1)),2),score(find(approachIndicesMS(1:idx1)),3),'.','Color',[1 0 0])
plot3(score(find(approachIndicesMS(idx1+1:idx2)),1),score(find(approachIndicesMS(idx1+1:idx2)),2),score(find(approachIndicesMS(idx1+1:idx2)),3),'.','Color',[.5 0 .4])

 title(['CRICKET/WALNUT coreg; app:red, eat:green'])        
 
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

clearvars sig_z
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

%% CHECK THE SAME -- MAHALANOBIS DISTANCE ACROSS ASSAYS -- USING THE ZSCORED DATA RATHER THAN PCA'ed

for mouseNum = 2:length(folders)
    
    eatScoreWalnut = sig_z_All{mouseNum}(find(eatIndicesMS_All{mouseNum}(1:idx1_All(mouseNum))),:);
    eatScoreCricket = sig_z_All{mouseNum}(find(eatIndicesMS_All{mouseNum}(idx1_All(mouseNum)+1:idx2_All(mouseNum))),:);

    appScoreWalnut = sig_z_All{mouseNum}(find(approachIndicesMS_All{mouseNum}(1:idx1_All(mouseNum))),:);
    appScoreCricket = sig_z_All{mouseNum}(find(approachIndicesMS_All{mouseNum}(idx1_All(mouseNum)+1:idx2_All(mouseNum))),:);
    
    distApp{mouseNum} = mahal(appScoreCricket,appScoreWalnut);

    distEat{mouseNum} = mahal(eatScoreCricket,eatScoreWalnut);

end

%Compile across mice
distApp_All = [];
distEat_All = [];

for mouseNum = 1:length(folders)
   distApp_All = [distApp_All; distApp{mouseNum}];
   distEat_All = [distEat_All; distEat{mouseNum}];   
end

meanDist = [mean(distApp_All),mean(distEat_All)];
seDist = [std(distApp_All)./sqrt(length(distApp_All)), std(distEat_All)./sqrt(length(distEat_All))];

figure(289)
bar(meanDist); hold on;
errorbar(meanDist,seDist,'LineStyle','none','Color','k')
ylabel('Mahalanobis dist. of cricket obs. from walnut ref.')
labels = {'approach','eat'};
set(gca, 'XTickLabel', labels)
[p,h, stats] = ranksum(distApp_All,distEat_All)
text(0,0,['p=' num2str(round(p,3))],'Color','r'); box off;
title(['n approach samples=' num2str(length(distApp_All)) ', n eat samples=' num2str(length(distEat_All))])