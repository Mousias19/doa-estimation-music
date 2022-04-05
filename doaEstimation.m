% A simple code to illustrate the operation of an OFDM transmitter and Receiver including 
% RF upconversion and down-conversion ; 
% References: IEEE802.11 standards, Simulation of Digital communication
% systems with MATLAB by Mathuranathan Viswanathan
clc; 
clear all;
%--------------------------------------------- 
%--------OFDM Parameters, an example of MCS0 of WLAN standard for a 20MHz wide channel --- 
%Modulation scheme used is BPSK
N = 64; %FFT size or total number of subcarriers (used + unused) 64 
N_s_data = 48; %Number of data subcarriers  
N_s_pilots = 4 ; %Number of pilot subcarriers  
ofdmBW = 20 * 10 ^ 6 ; % OFDM bandwidth 
%--------Derived Parameters-------------------- 
deltaF = ofdmBW/ N; %= 20 MHz/ 64 = 0.3125 MHz 
Tfft = 1/ deltaF; % IFFT/ FFT period = 3.2us 
Tgi = Tfft/ 4;% Guard interval duration and also the duration of cyclic prefix
Tsignal = Tgi + Tfft; %duration of BPSK-OFDM symbol 
Ncp = N* Tgi/ Tfft; %Number of symbols allocated to cyclic prefix 
Nst = N_s_data + N_s_pilots; %Number of total used subcarriers 
nBitsPerSym = Nst; %For BPSK the number of Bits per Symbol is same as num of subcarriers 
%-----------------Transmitter-------------------- 
s = 2*randi([0 1], 1, Nst)-1; %Generating Random Data with BPSK modulation 
% The number of bits being generated is limited to the total number of
% used sub-carriers. There could be more number of bits generated and the
% bits be re-arranged in a number of rows and columns equal to 'Nst' as
% in the commented code below
%data=randi([0 1], numbits, 1);  % Generate vector of of length 'numbits'
% s= [];
%for 0=0:Nst:numbits
%s1 = [data(i+1:(i+(Nst))]
% s = [ s; s1]
%IFFT block 
%Assigning subcarriers from 1 to 26 (mapped to 1-26 of IFFT input) 
%and -26 to -1 (mapped to 38 to 63 of IFFT input); 
%Nulls from 27 to 37 and at 0 position 
X_Freq =[ zeros( 1,1) s( 1: Nst/ 2) zeros( 1,11) s( Nst/ 2 + 1: end)]; 
% Assuming that the data is in frequency domain and converting to time domain 
% and scaling the amplitude appropriately
x_Time = N/ sqrt( Nst)* ifft( X_Freq); 
%Adding Cyclic Prefix 
ofdm_signal =[ x_Time( N-Ncp + 1: N) x_Time]; %Generation of the OFDM baseband signal complete
   
%%----------------------Up-conversion to RF----------------------------
Tsym = Tsignal/(N+Ncp); % duration of each symbol on the OFDM signal
t=Tsym/50:Tsym/50:Tsym; % define a time vector
%fc = 10*ofdmBW; % set the carrier frequency at 10 times the bandwidth of the OFDM channel
fc = 2.412 * 10 ^ 9; % set the carrier frequency at a WLAN cchannel's centre frequency
Carr_real=[];
Carr_imag=[];
Carr = [];
for n=1:length(ofdm_signal)
    Carr_real = real(ofdm_signal(n))*cos(2*pi*fc*t); %modulate the real part on a cosine carrier
    Carr_imag = imag(ofdm_signal(n))*sin(2*pi*fc*t); %modulate the imaginary part on a sine carrier
 %   Carr_real=[Carr_real Carr_real];
 %   Carr_imag=[Carr_imag Carr_imag];
    Carr = [Carr Carr_real+Carr_imag];
end
%% Addition of quantization noise. 
%The OFDM modulation operation is done in a Digital Signal processor in
%most cases whose digital output needs to be converted to analogue domain
%using a digital to analogue converter (DAC). The output of a DAC is
%quantized to discrete levels depending on the reference voltage and bit
%resolution of the DAC. This is generally done before up conversion and
%with individual DACs for the real and imaginary parts of the baseband
%signal. The quantization has been added after up-conversion in this code
%to avoid complexity of having two individual quantization codes for the
%real and imaginary parts of the base band. In either case, the output of
%the up-converter will have amplitudes at discrete staps depending on the
%quantization levels.
 n1 = 10; % Number of bits of the ADC/DAC
 max1= (2^n1)-1; %maximum n1 bit value
 m=length(Carr);
 Ac = max(abs(Carr));%Carrier amplitude of 1V
% %conversion of the signal ybb to n1 bits digital values quantized to the
% %nearest integer value
Vref = Ac*2; % Reference voltage of the converter
conv_fact1 = max1/Vref; %conversion scale for the analogue samples to convert to 16 bits
resolution = Vref/max1;
z1 = [];
for q=1:1:m
    z1(q)=(Carr(q)+Ac)*conv_fact1; %generating 'n1' bit digital representation 
                                          %of each sample of the carrier  
end
x1 = nearest(z1); % Each value is quantized to its nearest n1 bit number
y_tx = [];
for q=1:1:m
    y_tx(q) = ((x1(q)*Vref/max1)-Ac); %generating the analogue equivalent voltage of 
                                       %each 'n1' bit sample 
    qerr1(q) = Carr(q)-y_tx(q);
    
end
y_tx = Carr;
%NCarr = circshift(y_tx,delay);
%NCarr = delayseq(y_tx',delay)';
rx_carr = y_tx;
c = physconst('LightSpeed');
lam = c/fc;
         %  Element Spacing
            d = lam;
         %  Number of Elements   
            N = 4;
            theta = [35,25];
            beta = 2*pi;
            %beta=2*pi/wavelength; 
            phi=beta*(d/lam)*sin(theta*pi/180);
            M = length(theta);
         %  Steering Vectors
            for i=1:M
                for k=1:N
                    SteeringVector(k,i)= exp((k-1)*1i*phi(i));
                end
            end
            
%--------------Channel Modeling ----------------
% AWGN and other channel impairments could be added here. AWGN is added for
% inllustrative purpose. Further impairments such as doppler effect,
% Rayleigh fading may be added.
%snr = 200;
%rx_carr = awgn(rx_carr,snr, 'measured');
%%-----------------Receiver---------------------- 
%I-Q or vector down-conversion to recover the OFDM baseband signal from the
%modulated RF carrier
%O = exp(2*pi*-1i*2e-8*53*325000);
rx_carr = SteeringVector.*rx_carr;
r = [];
r_real= [];
r_imag = [];

snr = 15;
rx_carr = awgn(rx_carr,snr, 'measured');
for k=1:N
    for n=1:1:length(ofdm_signal)
 delay = 1e-7;
    %%XXXXXX inphase coherent dector XXXXXXX
    Z_in=rx_carr(k,(n-1)*length(t)+1:n*length(t)).*cos(2*pi*fc*(t)); %extract a period of the 
    %signal received and multiply the received signal with the cosine component of the carrier signal
    
    Z_in_intg=(trapz(t,Z_in(1,:)))*(2/Tsym);% integration using Trapizodial rule 
                                    %over a period of half bit duration
    r_real=Z_in_intg;
    
    %%XXXXXX Quadrature coherent dector XXXXXX
    Z_qd=rx_carr(k,(n-1)*length(t)+1:n*length(t)).*sin(2*pi*fc*(t));
    %above line indicat multiplication ofreceived & Quadphase carred signal
    
    Z_qd_intg=(trapz(t,Z_qd(1,:)))*(2/Tsym);%integration using trapizodial rule
        
    r_imag =  Z_qd_intg;   
        
    r=[r  r_real+1i*(r_imag)]; % Received Data vector
    end
end
for k = 1:4
           for l = 1:26
                arr(l,k) = exp(2*pi*-1i*l*deltaF*delay);
           end
           for l = 38:64
                arr(l-12,k) = exp(2*pi*-1i*l*deltaF*delay);
           end
end
r1 = reshape(r,[80,N]).';
%Removing cyclic prefix 
r_Parallel1 = r1(:,(Ncp + 1:(64 + Ncp))); 
%FFT Block 
r_Time(1,:) = sqrt(Nst)/ 64*(fft(r_Parallel1(1,:))); 
r_Time(2,:) = sqrt(Nst)/ 64*(fft(r_Parallel1(2,:))); 
r_Time(3,:) = sqrt(Nst)/ 64*(fft(r_Parallel1(3,:))); 
r_Time(4,:) = sqrt(Nst)/ 64*(fft(r_Parallel1(4,:))); 
%r_Time = sqrt(Nst)/ 64*(fft(r_Parallel(2,:)));
%Extracting the data carriers from the FFT output 
R_Freq1 = r_Time(:,[( 2: Nst/ 2 + 1) (Nst/ 2 + 13: Nst + 12)]).';
R_Freq1 = arr.*R_Freq1;
%s1 = circshift(s,1);
CSI(:,1)= R_Freq1(:,1)./s.';
CSI(:,2)= R_Freq1(:,2)./s.';
CSI(:,3)= R_Freq1(:,3)./s.';
CSI(:,4)= R_Freq1(:,4)./s.';
R_Freq = reshape(R_Freq1,1,[]);
CSI = reshape(CSI,1,[]);

%-------------------------------------------- 
%   MUSIC
    %Ap = 1;
    %B = exp(rand(1,1000)*2*pi*1i);
    %Rb = cov(B);
%   Auto-correlation matrix
    Rxx = CSI'*CSI/52;
%   eigenvalue decomposition
    [Vi,Li] = eig(Rxx);
%   sort in descending order
    [L,I] = sort(diag(Li),'descend');
    V = Vi(:,I);
%   Signal Subspace
    Ps = V(:,1:M)*V(:,1:M)';
%   Noise Subspace
    Pn = V(:,1+M:N)*V(:,1+M:N)';
    theta1=[0:90];
    tau=[0:1e-9:1e-6];
    %for l = 1:52
    % D(:,l) = l*deltaF;                
    % A = exp(-1i*2*pi*fc*tau(j));
    % B(:,l) = exp(-1i*2*pi*deltaF*l*(d/c*sin((i)*pi/180)*[0:N-1]'));
    % A1(:,l) = exp(-1i*2*pi*(fc+deltaF*l)*(d/c*sin((i)*pi/180)*[0:N-1]'+(tau(j))));
    % The MUSIC spectrum
    %end
    for i=1:length(theta1)
        phi1=2*pi*(d/lam)*sin(theta1(i)*pi/180);
        B =zeros([N 1]);
        for k=1:N
            B(k,1)= (exp((k-1)*1i*phi1));
        end
        B = B.';
       for j=1:length(tau)
           for l = 1:26
                A(:,l) = exp(2*pi*-1i*l*deltaF*tau(j));
           end
           for l = 38:64
                A(:,l-12) = exp(2*pi*-1i*l*deltaF*tau(j));
           end
            A1 = kron(B,A)';
%           PMUSIC(i,j) = real(A1'*A1)/real(A1'*Pn*A1);
%            PMUSIC(i,j)= ((A1*Pn).^2);
             PMUSIC(i,j)= 1./(sum(abs(A1'*Pn*A1)));
       end
    end
    

    [resSorted, orgInd] = sort(PMUSIC,'descend');
    DOAs = orgInd(1:M,1);
    
    [C,I] = max(PMUSIC(:));
    [I1,I2] = ind2sub(size(PMUSIC),I);
    deg = I1-1;
    del = I2-1;
%     figure(1);
%     plot(theta1,10*log10( PMUSIC));
%     title('MUSIC spectrum');
%     xlabel('Angle [degrees]');
%     ylabel('PMUSIC [dB]');   
    figure(1);
    [X,Y] = meshgrid(theta1,tau);
    surf(X,Y,(PMUSIC)')
    shading interp 
    colorbar    
    xlabel('Angle [degrees]');
    ylabel('Delay (s)');
    zlabel('PMUSIC');
    
    degrees = deg;
    delay = del;
