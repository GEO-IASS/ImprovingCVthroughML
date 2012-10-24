function b=labels_to_common_integers(a)
labs=getlablist(a);
n=size(a,1);
newlabels=zeros(n,1);
oldlabels=getlabels(a);
for i=1:size(labs,1)
    newlabels=newlabels+all(oldlabels==repmat(labs(i,:),n,1),2)*i;
end
b=setlabels(a,newlabels);
    
    