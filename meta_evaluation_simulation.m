function results=meta_evaluation_simulation(errors, descriptions, metac, crossvalidation)
% META_EVALUATION(accuracies, datacomplexity, metac, crossvalidation),
% evaluates a meta-problem using meta-classifier METAC and compare it to
% the performance of using cross validation to choose a classifier.
% 
% ERRORS is the n by c matrix containing errors for all n datasets
% and c classifiers. DESCRIPTIONS (n by k) contain the k meta-descriptors
% for all n datasets. METAC is the meta-classifier to be used (default
% knnc). CROSSVALIDATION (n by c) contains the cross validation error
% estimates for each c classifier.

narginchk(2, 4);
if (nargin==2)
    metac=knnc;
end

accuracies=errors;
datacomplexity=descriptions;

[g,i]=min(errors');
metaproblem=dataset(datacomplexity(:,:),i') % Form metaproblem
%scatterd(metaproblem*pca(metaproblem,2)) % Plot first two principal components of meta-problem

[metatrain,metatest,itrain,itest]=gendat(metaproblem(:,:),0.5); % Split metaproblem in a train and test set

results=[];

accuracies=accuracies(itest,:);

predicted=metatest*(metatrain*metac)
[u,l]=max(predicted');
predicted*testc;
results(1,1)=mean(l==i(itest));
results(1,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(1,3)=max(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
results(1,4)=max(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
results(1,5)=std(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));

% Predict the best meta-classifier
%mdescr=setname(setlabels(metatrain,num2str(metatrain.labels)),'Meta-Description')
%[u,l]=max(dataset(dcValues(mdescr))*(metatrain*metac))

%Worst choices
[u,l]=max(accuracies');
results(2,1)=mean(l==i(itest));
results(2,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(2,3)=sum(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));

%Best choices
[u,l]=min(accuracies');
results(3,1)=mean(l==i(itest));
results(3,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(3,3)=sum(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));

%Most common
l=mode(i(itrain))*ones(length(itest),1)';

results(6,1)=mean(l==i(itest));
results(6,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(6,3)=sum(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));

fprintf('Meta-learning (choice):\t %2.4f \n', results(1,1));
fprintf('Worst classifier:\t %2.4f \n', mean(max(accuracies'))); % Mean accuracy if you always choose the worst classifier
fprintf('Mean of  classifier:\t %2.4f \n', mean(mean(accuracies))); % Mean accuracy if you randomly choose either classifier
fprintf('Best of  classifier:\t %2.4f \n', mean(min(accuracies'))); % Mean accuracy if you always choose the best classifier

fprintf('Meta-learner:\t\t %2.4f \n', results(1,2)); % Mean accuracy using meta-learning
fprintf('Most common winner:\t %2.4f \n',mean(accuracies(:,mode(i(itrain))))) % Mean accuracy of most common winner in test set

fprintf('Error increase ML:\t %2.4f\n',results(1,3))
subplot(2,1,1); hist(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'),100);
title('Meta-errors')

%Cross-validation
[j,l]=min(crossvalidation(itest,:)');
results(4,1)=mean(i(itest)==l);
results(4,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(4,3)=sum(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
results(4,4)=max(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
results(4,5)=std(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
fprintf('Cross-validation (choice):\t %2.4f \n', results(4,1));
fprintf('Cross-validation:\t %2.4f \n',results(4,2)) % Mean accuracy using cross-validation
fprintf('Error increase CV:\t %2.4f\n',results(4,3))
subplot(2,1,2); hist(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'),100);

% Meta-learner with Cross-Validation descriptors
metaproblem=dataset(crossvalidation(:,:), i')
metatrain=metaproblem(itrain,:);
metatest=metaproblem(itest,:);

if (size(metaproblem,2)==2)
    subplot(3,3,[1,2,4,5,7,8])
    coll=hsv
    %scatterd(metaproblem(itest(itest<=500),:),2,coll(1:4,1:3))
    plot(+metaproblem(itest(itest<=500 & +metaproblem.labels(itest)==1),1),+metaproblem(itest(itest<=500 & +metaproblem.labels(itest)==1),2), 'go')
    hold on
    plot(+metaproblem(itest(itest<=500 & +metaproblem.labels(itest)==2),1),+metaproblem(itest(itest<=500 & +metaproblem.labels(itest)==2),2), 'mo')
    
    %scatterd(metaproblem(itest(itest>500),:),2,coll(1:4,1:3))
    %plot(+metaproblem(itest(itest>500),1), +metaproblem(itest(itest>500),2), 'g+')
    plot(+metaproblem(itest(itest>500 & +metaproblem.labels(itest)==1),1),+metaproblem(itest(itest>500 & +metaproblem.labels(itest)==1),2), 'g+')
    %hold on
    plot(+metaproblem(itest(itest>500 & +metaproblem.labels(itest)==2),1),+metaproblem(itest(itest>500 & +metaproblem.labels(itest)==2),2), 'm+')
    legend('  NM, G','1-NN, G','  NM, B','1-NN, B','Location','Best')
    %legend('NM best choice','1-NN best choice')
    title('Meta-problem')
    ylabel('10-fold CV error 1-NN')
    xlabel('10-fold CV error Nearest Mean')
    axis([0.0 0.7 0.0 0.7])
    plotc(metatrain*metac,1,'k--');
    hold on; plot([0, 1],[0,1],'k')
    hold off
    subplot(3,3,3)
    scatterd(gendats([1000 1000],2,1))
    axis off
    title('Gaussian problem')
    subplot(3,3,6)
    scatterd(gendatb([1000 1000],0.5))
    axis off
    subplot(3,3,9)
    scatterd(gendatd([1000 1000],100,4,0))
    axis off
end

predicted=metatest*(metatrain*metac);
[u,l]=max(predicted');

results(5,1)=mean(i(itest)==l);
results(5,2)=mean(accuracies(sub2ind(size(accuracies),[1:length(l)],l)));
results(5,3)=sum(accuracies(sub2ind(size(accuracies),[1:length(l)],l))-min(accuracies'));
fprintf('Trained CV (choice):\t %2.4f \n', results(5,1));
fprintf('Trained cross-validation:\t %2.4f \n',results(5,2));

% Predict the best meta-classifier
%mdescr=setname(setlabels(metatrain,num2str(metatrain.labels)),'Meta-Description')
%descr=dataset(crossval(metatrain,{nmc,knnc},10));
%[u,l]=max(descr*(metatrain*metac))


results

%scatterd(dataset(cvalues(:,2:3),i'))