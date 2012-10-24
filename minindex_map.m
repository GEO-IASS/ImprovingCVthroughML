function F = minindex_map(T,W)
	prtrace(mfilename);
    n=size(+T,1);
    m=size(+T,2);
    [g,i]=min(+T,[],2);
    li=sub2ind(size(+T), 1:n, i');
    F=zeros(n,m);
    F(li)=1;
    F=setdata(T,F,getlabels(W));
return;

