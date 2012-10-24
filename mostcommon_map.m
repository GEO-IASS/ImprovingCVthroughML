function F = mostcommon_map(T,W)
	prtrace(mfilename);
    n=size(+T,1);
    m=size(+T,2);
    c=length(getlablist(T));
    F=zeros(n,c);
    data = getdata(W);
    data=find(getlablist(T)==data);
    F(:,data)=ones(n,1);
    F=setdata(T,F,getlabels(W)); %Turn the result into dataset
return;