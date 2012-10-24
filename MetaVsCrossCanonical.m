% The clearest case of meta-learning outperforming cross-validation
% selection, used as experiment 1 in the paper

%% Settings
%classifiers={nmc,ldc,knnc,svc,treec,parzenc}
clear
classifiers={knnc([],1), nmc, fisherc}
nproblems=1000;
nfolds=5;
dataset_directory='~/Code/Complexity Transformations/S1/'

prwarning off
prwaitbar off

%% Generate data

pr=1
% Gaussian problem
for i=1:(nproblems/2)
    problem=gendats([10000 10000],2,rand);
    trn_n=randi([20 100]);
    [train,test]=gendat(problem,trn_n);
    [e_cv,std_cv]=crossval(train,classifiers,10,10);
    crossvalidation(pr,:)=e_cv;
    errors(pr,:)=cell2mat(testc(test,train*classifiers));
    metafeatures(pr,:)=[size(train, 1) std_cv];
    sourcelabels(pr)=1;
    pr=pr+1
end

% Banana set problem
for i=1:(nproblems/2)
    problem=gendatb([10000 10000],rand*2);
    trn_n=randi([20 100]);
    [train,test]=gendat(problem,trn_n);
    [e_cv,std_cv]=crossval(train,classifiers,10,10);
    crossvalidation(pr,:)=e_cv;
    errors(pr,:)=cell2mat(testc(test,train*classifiers));
    metafeatures(pr,:)=[size(train, 1) std_cv];
    sourcelabels(pr)=1;
    pr=pr+1
end

%% Construct metaproblem
[g,i]=min(errors(:,1:2)');
metaproblem=dataset(crossvalidation(:,1:2),i');
metaproblem=addlabels(metaproblem,sourcelabels','source')
metaproblem=changelablist(metaproblem,'default');

[g,i]=min(errors(:,1:2)');
metaproblemextended=dataset([crossvalidation(:,1:2) metafeatures(:,1:2)],i');
metaproblemextended=addlabels(metaproblemextended,sourcelabels','source')
metaproblemextended=changelablist(metaproblemextended,'default');

%% Train and evaluate meta-learners
crossval(metaproblem, {mostcommonc,minindexc,ldc,svc},10)
crossval(metaproblemextended, {ldc,svc},10)

%% Visualize
figure
gridsize(500)
subplot(3,3, [1 2 4 5 7 8])
metaproblem=changelablist(metaproblem,'source')
sources=getlabels(metaproblem);
metaproblem=changelablist(metaproblem,'default')
plot(+metaproblem((sources==1 & getlabels(metaproblem)==1),1),+metaproblem((sources==1 & getlabels(metaproblem)==1),2), 'go')
hold on
plot(+metaproblem((sources==1 & getlabels(metaproblem)==2),1),+metaproblem((sources==1 & getlabels(metaproblem)==2),2), 'mo')
plot(+metaproblem((sources==2 & getlabels(metaproblem)==1),1),+metaproblem((sources==2 & getlabels(metaproblem)==1),2), 'g+')
plot(+metaproblem((sources==2 & getlabels(metaproblem)==2),1),+metaproblem((sources==2 & getlabels(metaproblem)==2),2), 'm+')
legend('  NM, G','1-NN, G','  NM, B','1-NN, B','Location','Best')
plotc(svc(metaproblem),1.5)
plotc(minindexc(metaproblem),1,'k-.')
title('2 base-type meta-problem')
ylabel('10-fold CV error 1-NN')
xlabel('10-fold CV error Nearest Mean')




axis([0.0 0.7 0.0 0.7])
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