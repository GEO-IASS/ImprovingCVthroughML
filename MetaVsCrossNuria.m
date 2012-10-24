% Test the effect of meta-learning vs. ordinary cross-validation on Macia's 300 datasets. 
% These sets are supposed to span the data complexity space, measuring different types of difficulties in classification that we can come across

%% Settings
%classifiers={nmc,ldc,knnc,svc,treec,parzenc} # Small set used to determine
% best sample size.
clear
classifiers={robustveil([],nmc), robustveil([],knnc), robustveil([],fisherc), robustveil([],qdc), robustveil([],parzenc), robustveil([],stumpc),robustveil([],svc),robustveil([],svc([],proxm([],'r')))}
%classifiers={robustveil([],nmc), robustveil([],knnc([],1)), robustveil([],knnc), robustveil([],fisherc),robustveil([],qdc)}
nproblems=300;
nfolds=5;
dataset_directory='~/Code/Complexity Transformations/S1/' % Directory with the 300 datasets
trn_n=50
type='density'

prwarning off
prwaitbar off

%% Multiple folds: Run the main loop (Not used)
% pr=0;
% for p=1:nproblems
%     problem=prarff(strcat(dataset_directory, 'D',num2str(p),'-trn.arff'));
%     problem=setprior(problem,getprior(problem)); % Circumvent error messages
%     shuffled=problem(randperm(size(problem,1)),:); % Shuffled the objects in the dataset
%     
%     % Split the dataset up into different folds
%     for f=1:nfolds
%         assignment=mod(1:size(problem,1),nfolds)+1;
%         fold{f}=problem(assignment==f,:);
%     end
%     
%     % Treat each fold as a separate problem with a large test set
%     for f=1:nfolds
%         fprintf(strcat('Problem ',num2str(p),', Fold ',num2str(f),' \n'))
%         pr=pr+1;
%         
%         testfolds=1:nfolds;
%         testfolds(f)=[];
%         test=fold{testfolds(1)};
%         for i=2:length(testfolds)
%             test=[test; fold{testfolds(i)}];
%         end
%         train=fold{f};
%      
%         
%         crossvalidation(pr,:)=crossval(train,classifiers,10);
%         errors(pr,:)=cell2mat(testc(test, train*classifiers));
%         meta_features(pr,:)=size(train);
%         sourcelabels(pr)=p;
%     end
% end
% clear p i f problem shuffled test testfolds train fold

%% Calculate the features for the meta-problem
pr=0;
t=0
for p=1:nproblems
    problem=prarff(strcat(dataset_directory, 'D',num2str(p),'-trn.arff'));
    problem=labels_to_common_integers(problem);
    problem=setprior(problem,getprior(problem)); % Circumvent error messages
    shuffled=problem(randperm(size(problem,1)),:); % Shuffled the objects in the dataset
    
    
    % Treat each fold as a separate problem with a large test set
    for f=1:nfolds
        tic;
        pr=pr+1
        if strcmp(type,'density')
            generated_problem=gendatp(problem,20000)
            [train,test]=gendat(generated_problem,trn_n)
        elseif strcmp(type,'subsampling')
            display('Using subsampling')
            [train,test]=gendat(problem,trn_n)
        else
            error('Not a correct data generation type')
        end
            
        % Reset priors so we do not assume accurate estimation of the class
        % priors
        train=setprior(train,[]);
        train=setprior(train,getprior(train));
        test=setprior(test,[]);
        test=setprior(test,getprior(test));
        
        [e_cv,std_cv]=crossval(train,classifiers,10,5,testd);
        crossvalidation(pr,:)=e_cv;
        resubstitution(pr,:)=cell2mat(testc(train, train*classifiers));
        
        errors(pr,:)=cell2mat(testc(test, train*classifiers));
        %errors2(pr,:)=cell2mat(testc2(test, train*classifiers));
        meta_features(pr,:)=std_cv;
        sourcelabels(pr)=p;
        toc
    end
end
clear p i f problem shuffled test testfolds train fold

%% Evaluation (Deprecated)
meta_evaluation(errors(:,1:2), meta_features, knnc, crossvalidation2(:,1:2))

%% Build metaproblem with classes reversed
[g,i]=min(errors(~isnan(crossvalidation(:,1)),2:-1:1)');
metaproblem=dataset(crossvalidation(~isnan(crossvalidation(:,1)),2:-1:1),i');
metaproblem=addlabels(metaproblem,sourcelabels(~isnan(crossvalidation(:,1)))','source')
metaproblem=changelablist(metaproblem,'default');

%% Build metaproblem and extended metaproblem including extra features
[g,i]=min(errors(:,:)');
metaproblem=dataset(crossvalidation(:,:),i');
metaproblem=addlabels(metaproblem,sourcelabels','source')
metaproblem=changelablist(metaproblem,'default');

[g,i]=min(errors(:,:)');
metaproblemextended=dataset([crossvalidation(:,:) meta_features(:,:)],i');
metaproblemextended=addlabels(metaproblemextended,sourcelabels','source')
metaproblemextended=changelablist(metaproblemextended,'default');

%% Generate Images
[g,i]=min(errors(:,[2 3])');
metaproblem=dataset(crossvalidation(:,[2 3]),i')
metaproblem=addlabels(metaproblem,sourcelabels','source')
metaproblem=changelablist(metaproblem,'default');
crossval(metaproblem, {mostcommonc,minindexc,ldc,knnc},10)
crossval(metaproblemextended, {ldc,knnc},10)

figure; 
metaproblem=changelablist(metaproblem(1:100,:),'source'); 
scatterd(metaproblem);
title('Real-world data meta-problem')
ylabel('10-fold CV error 1-NN')
xlabel('10-fold CV error Nearest Mean')
gridsize(500)
plotc(minindexc(metaproblem),1,'k-.')
metaproblem=changelablist(metaproblem,'default');


[g,i]=min(errors(:,[3 5])');
metaproblem=dataset(crossvalidation(:,[3 5]),i')
metaproblem=addlabels(metaproblem,sourcelabels','source')
metaproblem=changelablist(metaproblem,'default');
figure
gridsize(500)
scatterd(metaproblem)
plotc(ldc(metaproblem),1.5)
plotc(minindexc(metaproblem),1,'k-.')
title('Real-world data meta-problem')
ylabel('10-fold CV error Parzen')
xlabel('10-fold CV error Fisher')
legend('Fisher Best','Parzen Best','Location','SouthEast')

[e,c,d]=loso(metaproblem, knnc, 'source')
[e,c,d]=loso(metaproblem, ldc, 'source')
loso(metaproblem, minindexc, 'source')
loso(metaproblem, mostcommonc, 'source')
loso(metaproblemextended, ldc, 'source')
[e,c,d]=loso(metaproblemextended, knnc, 'source')

%% Difference in accuracy
[err,cerr,lab1]=crossval(metaproblem, {minindexc},10)
[err,cerr,lab2]=crossval(metaproblemextended, {knnc},10)

mean(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))
mean(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))
mean(max(errors')-min(errors'))

var(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'))
var(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'))
var(max(errors')-min(errors'))

figure
subplot(1,2,1)
hist(errors(sub2ind(size(errors),[1:length(lab1{1})],lab1{1}'))-min(errors'),100)
subplot(1,2,2)
hist(errors(sub2ind(size(errors),[1:length(lab2{1})],lab2{1}'))-min(errors'),100)

%% Evaluate different metatrain sizes
selectproblems=gendat(metaproblem,0.02)
loso(selectproblems,minindexc,'source')
loso(selectproblems,ldc,'source')