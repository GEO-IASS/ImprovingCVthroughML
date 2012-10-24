% The clearest case of meta-learning outperforming cross-validation
% selection, used as experiment 1 in the paper

%% Settings
%classifiers={nmc,ldc,knnc,svc,treec,parzenc}
clear
classifiers={robustveil([],nmc), robustveil([],fisherc)}
nproblems=1000;
nfolds=5;
dataset_directory='~/Code/Complexity Transformations/S1/'
trn_n=40
prwarning off
prwaitbar off

%% Generate data

pr=1
for i=1:(nproblems)
    
    smp = randi(11)+19;
    p1 = 10/smp + (rand^2)*(smp-20)/smp;    
    problem = gendatd([10000 10000],100);
    [train,test]=gendat(problem,smp);
    [e_cv,std_cv]=crossval(train,classifiers,10,10);
    crossvalidation(pr,:)=e_cv;
    errors(pr,:)=cell2mat(testc(test,train*classifiers));
    metafeatures(pr,:)=[size(train, 1) std_cv];
    sourcelabels(pr)=1;
    pr=pr+1
end


%% Construct metaproblem
[g,i]=min(errors(:,:)');
metaproblem=dataset(crossvalidation(:,:),i');
metaproblem=addlabels(metaproblem,sourcelabels','source')
metaproblem=changelablist(metaproblem,'default');

[g,i]=min(errors(:,1:2)');
metaproblemextended1=dataset([crossvalidation(:,:) metafeatures(:,2:end)],i');
metaproblemextended2=dataset([crossvalidation(:,:) metafeatures(:,1)],i');
metaproblemextended3=dataset([crossvalidation(:,:) metafeatures(:,:)],i');

%% Train and evaluate meta-learners
crossval(metaproblem, {mostcommonc,minindexc,ldc,knnc,scalem([],'variance')*knnc},10,10)
crossval(metaproblemextended1, {ldc,scalem([],'variance')*knnc},10,10)
crossval(metaproblemextended2, {ldc,scalem([],'variance')*knnc},10,10)
crossval(metaproblemextended3, {ldc,scalem([],'variance')*knnc},10,10)
%crossval(metaproblem, {minindexc,ldc,svc},10)

%% Difference in accuracy
[err,cerr,lab1]=crossval(metaproblem, {minindexc},10)
[err,cerr,lab2]=crossval(metaproblemextended3, {ldc},10)

mean((errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))./min(errors'))
mean((errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))./min(errors'))
mean((max(errors')-min(errors'))./min(errors'))

mean(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))
mean(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))
mean(max(errors')-min(errors'))

var(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))
var(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))
var(max(errors')-min(errors'))

max(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))
max(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))
max(max(errors')-min(errors'))

figure
subplot(1,2,1)
hist(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'),100)
subplot(1,2,2)
hist(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'),100)


%% Visualize
figure
gridsize(500)
subplot(3,3
scatterd(metaproblem)
plotc(ldc(metaproblem),1.5)
plotc(minindexc(metaproblem),1,'k-.')
title('Real-world data meta-problem')
ylabel('10-fold CV error 1-NN')
xlabel('10-fold CV error Nearest Mean')