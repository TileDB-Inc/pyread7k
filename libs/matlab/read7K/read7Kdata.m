%% Load data
try
    load pathname
catch
    pathname = [];
end
[filename, pathname] = uigetfile([pathname '*.s7k'], 'Pick an s7k-file');
fid = fopen([pathname filename]);
if pathname ~= 0
    save pathname pathname
end

%% Initialize
ping    = 0;
maxPing = 10; % break while loop when 'maxPing' is reached
fig     = figure;


%% read data
while true
    [header, data] = read7Krecord(fid);
    
%   disp(['Record type:  ' num2str(header.recordType)]);
    
    if isempty(header.recordType)
        break;
    end
    if header.size == 0
        break;
    end
    
    if isfield(header, 'ping_number')
        ping_number = header.ping_number;
        disp(['Ping #  ' num2str(header.pingNumber)]);
    end

    % get various parameters
    if header.recordType == 7000
        F   = header.frequency/1000; % kHz
        T   = header.txPulseWidth;
        Fs  = header.sampleRate;
        c   = header.soundVelocity;
        if header.txPulseType == 0
            pulsType = 'CW';
        else
            pulsType = 'FM';
        end
        absorption  = header.absorption;
        spreading = header.spreadingLoss;
        rangeSetting = header.range;
    end
    
    % beam angles
    if header.recordType == 7004
        tet = 180/pi*data.beamHorizontalDirection;
        tet = tet';
    end
    
    % beam formed data
    if header.recordType == 7018
        ping = ping + 1;

        figure(fig), 
        subplot 121, 
        imagesc(data.mag)
        title('Amplitude')
        xlabel('Channel number')
        ylabel('Range [samples]')
        
        subplot 122, 
        imagesc(data.phase*(1/2^15*pi)) % scale phase to ±pi
        title('Phase')
        xlabel('Channel number')
        ylabel('Range [samples]')
        
    end
    
    % raw channel data
    if header.recordType == 7038
        ping = ping + 1;
        
        figure(fig), 
        subplot 121, 
        imagesc(sqrt(real(data.value).^2 + imag(data.value).^2))
        title('Amplitude')
        xlabel('Channel number')
        ylabel('Range [samples]')
        
        subplot 122, 
        imagesc(atan2(imag(data.value), real(data.value)))
        title('Phase')
        xlabel('Channel number')
        ylabel('Range [samples]')

    end
    
    
    if ping == maxPing
        fclose all;
        break;
    end
    
end