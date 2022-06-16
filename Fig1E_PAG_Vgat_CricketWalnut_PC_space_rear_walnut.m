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

assayNum = 1; %1=walnut, 2=cricket

%%

for mouseNum = 1:size(folders,1)

cd(folders{mouseNum,assayNum})
load('output_CNMF-E.mat','neuron')
load('good_neurons.mat')

sig = neuron.C_raw(find(good_neurons),:);
sig = sig';

        if assayNum == 1
            %load('Seg.mat','behavSegAll')
            load('eating_vars.mat','detect_indice','eating_indice')
            approachIndicesMS = detect_indice; eatIndicesMS = eating_indice; clearvars detect_indice eating_indice
            load('BehaviorMS_Rear.mat','rearingIndicesMS')
        elseif assayNum == 2
            %load('Seg_ManuallyChecked.mat','behavSegAll')
            load('BehaviorMS_2.mat','approachIndicesMS','eatIndicesMS')
            load('BehaviorMS_Rear.mat','rearingIndicesMS')
        end

        if assayNum==1
            fracWalnut = .26; %fraction of session post walnut introduction.
            sessLength = length(neuron.C_raw);
            OF_Indices = 1:round(fracWalnut .* sessLength); %first 15% of session is open field.
            Walnut_Indices = (round(fracWalnut .* sessLength)):sessLength;
            
            sig = sig(Walnut_Indices(1):Walnut_Indices(end),:); %remove artificial prey data.
            %mouseVel = mouseVel(Walnut_Indices(1):Walnut_Indices(end));
            approachIndicesMS = approachIndicesMS(Walnut_Indices(1):Walnut_Indices(end));
            eatIndicesMS = eatIndicesMS(Walnut_Indices(1):Walnut_Indices(end)); 
            rearingIndicesMS = rearingIndicesMS(Walnut_Indices(1):Walnut_Indices(end));            
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

%plot Open Field timepoints
%plot3(score(OF_Indices,1),score(OF_Indices,2),score(OF_Indices,3),'.','Color','b')

%plot Walking timepoints
%plot3(score(find(walkingIndices),1),score(find(walkingIndices),2),score(find(walkingIndices),3),'.','Color','b')

    if assayNum == 2
        title(['app crick:red, eat crick:green, rear:blue'])        
    elseif assayNum == 1
        title(['approach walnut: red, eat walnut:green'])
    end
 
%CALCULTE THE SILHOUETTE SCORE FOR THE CLUSTERS:
clusterID = nan(length(score),1);
clusterID(find(approachIndicesMS)) = 1;
clusterID(find(eatIndicesMS)) = 2;
%clusterID(find(rearingIndicesMS)) = 3;
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

end

%%
figure(57)
meanS = nanmean(silhouetteScore);
seS = nanstd(silhouetteScore) ./ sqrt(size(folders,1));
bar(meanS); hold on;
errorbar(meanS,seS,'LineStyle','none','Color','k'); hold on;
scatter(ones(1,size(folders,1)),silhouetteScore,10,'filled')
ylim([0 .8])
ylabel('silhouette scores')
[h,p,~,stats] = ttest(silhouetteScore)
tstatVal = stats.tstat;
title(['p=' num2str(round(p,3)) ' tstat=' num2str(tstatVal)],'Color','r')