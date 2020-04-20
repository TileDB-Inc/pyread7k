function [header, data] = read7Krecord(fid)

% just for reference:
% BYTE ~ unsigned char: 8 bit
% WORD ~ unsigned short int:  16 bit
% DWORD ~unsigned int: 32 bit
%

isNewFormat = 1;

while fread(fid, 1, 'uint32') ~= 65535 % read until sync pattern detected 
end

% header.protocolVersion          = fread(fid, 1, 'uint16'); % 2 bytes,	version of the frame protocol, 1, 2, 3, or 4
% header.offset                   = fread(fid, 1, 'uint16'); % 2 bytes,	offest in bytes from start of sync pattern to start of data section
% header.syncPattern              = fread(fid, 1, 'uint32'); % 4 bytes,
header.size                     = fread(fid, 1, 'uint32'); % 4 bytes,
header.optionalDataOffset       = fread(fid, 1, 'uint32'); % 4 bytes,
header.optionalDataIdentifier   = fread(fid, 1, 'uint32'); % 4 bytes,

header.year                     = fread(fid, 1, 'uint16'); % 2 bytes,
header.day                      = fread(fid, 1, 'uint16'); % 2 bytes,
header.seconds                  = fread(fid, 1, 'float'); % 4 bytes,
header.hours                    = fread(fid, 1, 'uint8'); % 1 byte,
header.minutes                  = fread(fid, 1, 'uint8'); % 1 byte,

header.reserved1                = fread(fid, 1, 'uint16'); % 2 bytes,
header.recordType               = fread(fid, 1, 'uint32'); % 4 bytes,
header.deviceIdentifier         = fread(fid, 1, 'uint32'); % 4 bytes,
header.reserved2                = fread(fid, 1, 'uint16'); % 2 bytes,
header.systemEnumerator         = fread(fid, 1, 'uint16'); % 2 bytes,
header.reserved3                = fread(fid, 1, 'uint32'); % 4 bytes,
header.drfFlags                 = fread(fid, 1, 'uint16'); % 2 bytes,
header.reserved4                = fread(fid, 1, 'uint16'); % 2 bytes,
header.reserved5                = fread(fid, 1, 'uint32'); % 4 bytes,
header.totalRecords             = fread(fid, 1, 'uint32'); % 4 bytes, total number of records in fragmented data record set, if appropriate flag is set
header.fragmentNumber           = fread(fid, 1, 'uint32'); % 4 bytes,

data = [];
% -------------------------------------------------------------
%                                  TOTAL HEADER SIZE = 64 bytes
% -------------------------------------------------------------

if header.recordType == 7000
    header.serialNumber                         = fread(fid, 1, 'uint64'); % 8 bytes
    header.pingNumber                           = fread(fid, 1, 'uint32'); % 4 bytes
    header.multipingSequence                    = fread(fid, 1, 'uint16'); % 4 bytes
    header.frequency                            = fread(fid, 1, 'float'); % 4 bytes
    header.sampleRate                           = fread(fid, 1, 'float'); % 4 bytes
    header.receiverBandwidth                    = fread(fid, 1, 'float'); % 4 bytes
    header.txPulseWidth                         = fread(fid, 1, 'float'); % 4 bytes
    header.txPulseType                          = fread(fid, 1, 'uint32'); % 4 bytes
    header.txPulseEnvelopeType                  = fread(fid, 1, 'uint32'); % 4 bytes
    header.txPulseEnvelopeParameter             = fread(fid, 1, 'float'); % 4 bytes
    header.txPulseReserved                      = fread(fid, 1, 'uint32'); % 4 bytes
    header.maxPingRate                          = fread(fid, 1, 'float'); % 4 bytes
    header.pingPeriod                           = fread(fid, 1, 'float'); % 4 bytes
    header.range                                = fread(fid, 1, 'float'); % 4 bytes
    header.power                                = fread(fid, 1, 'float'); % 4 bytes
    header.gain                                 = fread(fid, 1, 'float'); % 4 bytes
    header.controlFlags                         = fread(fid, 1, 'uint32'); % 4 bytes
    header.projectorMagicNumber                 = fread(fid, 1, 'uint32'); % 4 bytes
    header.projectorBeamSteerX                  = fread(fid, 1, 'float'); % 4 bytes
    header.projectorBeamSteerZ                  = fread(fid, 1, 'float'); % 4 bytes
    header.projectorBeamWidthX                  = fread(fid, 1, 'float'); % 4 bytes
    header.projectorBeamWidthZ                  = fread(fid, 1, 'float'); % 4 bytes
    header.projectorBeamFocalPoint              = fread(fid, 1, 'float'); % 4 bytes
    header.projectorBeamWeightingType           = fread(fid, 1, 'uint32'); % 4 bytes
    header.projectorBeamWeightingParameter      = fread(fid, 1, 'float'); % 4 bytes
    header.transmitFlags                        = fread(fid, 1, 'uint32'); % 4 bytes
    header.hydrophoneMagicNumber                = fread(fid, 1, 'uint32'); % 4 bytes
    header.hydrophoneBeamWeightingType          = fread(fid, 1, 'uint32'); % 4 bytes
    header.hydrophoneBeamWeightingParameter     = fread(fid, 1, 'float'); % 4 bytes933
    header.receiveFlags                         = fread(fid, 1, 'uint32'); % 4 bytes
    header.receiveBeamWidth                     = fread(fid, 1, 'float'); % 4 bytes
    header.bottomDetectionInfo1                 = fread(fid, 1, 'float'); % 4 bytes	% Filter: Range Min input
    header.bottomDetectionInfo2                 = fread(fid, 1, 'float'); % 4 bytes	% Filter: Range Min input
    header.bottomDetectionInfo3                 = fread(fid, 1, 'float'); % 4 bytes	% Filter: Depth Min input
    header.bottomDetectionInfo4                 = fread(fid, 1, 'float'); % 4 bytes	% Filter: Depth Max input
    header.absorption                           = fread(fid, 1, 'float'); % 4 bytes
    header.soundVelocity                        = fread(fid, 1, 'float'); % 4 bytes
    header.spreadingLoss                        = fread(fid, 1, 'float'); % 4 bytes
    header.reserved                             = fread(fid, 1, 'uint16'); % 2 bytes
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 1003 % position
	header.datumId           = fread(fid, 1, 'uint32'); % 4 bytes, DWORD
	header.latency            = fread(fid, 1, 'float');  % 4 bytes, float
	header.latitudeNorthing  = fread(fid, 1, 'double'); % 8 bytes, double
	header.longitudeEasting  = fread(fid, 1, 'double'); % 8 bytes, double
	header.height             = fread(fid, 1, 'double'); % 8 bytes, double
	header.positionTypeFlag = fread(fid, 1, 'uint8');  % 1 bytes, BYTE
	header.utmZone           = fread(fid, 1, 'uint8');  % 1 bytes, BYTE
	header.qualityFlag       = fread(fid, 1, 'uint8');  % 1 bytes, BYTE
	header.positionMethod    = fread(fid, 1, 'uint8');  % 1 bytes, BYTE
    
    if header.size-64-4-36 > 0
        header.nSattelites = fread(fid, 1, 'uint8');  % 1 bytes, BYTE
    end
    
    header.checksum           = fread(fid, 1, 'uint32'); % 2 bytes

   
elseif header.recordType == 1012
    header.roll               = fread(fid, 1, 'float');
	header.pitch              = fread(fid, 1, 'float');
	header.heave              = fread(fid, 1, 'float');

    header.checksum           = fread(fid, 1, 'uint32'); % 2 bytes
   
elseif header.recordType == 1013% heading
    header.heading            = fread(fid, 1, 'float');
    
    header.checksum           = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 1015% heading
    header.verticalReference        = fread(fid, 1, 'uint8');
    header.latitude                 = fread(fid, 1, 'float64');
    header.longitude                = fread(fid, 1, 'float64');
    header.horizontalPosAccuracy    = fread(fid, 1, 'float32');
    header.vesselHeight             = fread(fid, 1, 'float32');
    header.heightAccuracy           = fread(fid, 1, 'float32');
    header.speedOverGround          = fread(fid, 1, 'float32');
    header.courseOverGround         = fread(fid, 1, 'float32');
    header.heading                  = fread(fid, 1, 'float32');
    
    header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7001
    header.sonarId = fread( fid, 1, 'uint64' );
    header.noSonars = fread( fid, 1, 'uint32' );
    for nSonar = 1 : header.noSonars
        data(nSonar).id                         = fread( fid, 1, 'uint32' );
        data(nSonar).description                = char( fread( fid, 64, 'char' ))';
        data(nSonar).serialNumber               = fread( fid, 1, 'uint64' );
        data(nSonar).infoLength                 = fread( fid, 1, 'uint32' );
        xmlString                               = fread( fid, data(nSonar).infoLength, 'char' );
        %xmlString                              = xmlString(( xmlString ~= 10 ) & ( xmlString ~= 13 ));
        data(nSonar).info                       = char( xmlString )';
    end
    
    header.checksum           = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7002
    bytesRead = 64;
    
    header.sonarId                              = fread( fid, 1, 'uint64' ); bytesRead = bytesRead + 8;
    header.pingNumber                           = fread( fid, 1, 'uint32' ); bytesRead = bytesRead + 4;
    header.operation                            = fread( fid, 1, 'uint32' ); bytesRead = bytesRead + 4;
    header.startFrequency                       = fread( fid, 1, 'single' ); bytesRead = bytesRead + 1;
    header.stopFrequency                        = fread( fid, 1, 'single' ); bytesRead = bytesRead + 1;
    header.windowType                           = fread( fid, 1, 'uint32' ); bytesRead = bytesRead + 4;
    
    bytesNotRead = header.size - bytesRead;
    if bytesNotRead
        fread(fid, bytesNotRead-6, 'uint8'); 
%         disp(['Record 7200: Read ' num2str(bytesNotRead) ' Extra bytes']);
    end
    
    % NEXT 3 LINES CAUSED PROBLERMS IN SOME S7K FILES, WHERE RECORD SIZE WOULD BE EXCEEDED
%   header.shadingValue                         = fread( fid, 1, 'float' ); bytesRead = bytesRead + 4;
%   header.reservedArray                        = fread( fid, 14, 'uint32' );
%   header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 7004 % 7k Beam Geometry
    header.sonarId                              = fread(fid, 1, 'uint64');
    header.receiverBeams                        = fread(fid, 1, 'uint32');
    
    data.beamVerticalDirection                  = fread( fid, header.receiverBeams, 'float' );
    data.beamHorizontalDirection                = fread( fid, header.receiverBeams, 'float' );
    data.beamWidthX                             = fread( fid, header.receiverBeams, 'float' );
    data.beamWidthY                             = fread( fid, header.receiverBeams, 'float' );
    header.checksum                             = fread(fid, 1, 'uint32');

    
elseif header.recordType == 7006 % BATHYMETRIC DATA
    header.sonarId                              = fread(fid, 1, 'uint64'); % 8 bytes
    header.pingNumber                           = fread(fid, 1, 'uint32'); % 4 bytes
    header.multiPingSequence                    = fread(fid, 1, 'uint16'); % 2 bytes
    header.N                                    = fread(fid, 1, 'uint32'); % 4 bytes
    header.layerCompensationFlag                = fread(fid, 1, 'uint8'); % 1 byte
    header.soundVelocityFlags                   = fread(fid, 1, 'uint8'); % 1 byte
    header.soundVelocity                        = fread(fid, 1, 'float'); % 4 bytes
    
    data.range                                  = fread(fid, header.N, 'float'); % 4 bytes
    data.quality                                = fread(fid, header.N, 'uint8'); % 4 bytes
    data.intensity                              = fread(fid, header.N, 'float'); % 4 bytes
    data.minFilterInfo                          = fread(fid, header.N, 'float'); % 4 bytes
    data.maxFilterInfo                          = fread(fid, header.N, 'float'); % 4 bytes
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 7007
    header.sonarId                          = fread(fid, 1, 'uint64');
    header.pingNumber                       = fread(fid, 1, 'uint32');
    header.multiPingSequence                = fread(fid, 1, 'uint16');
    header.beamPosition                     = fread(fid, 1, 'float');
    header.controlFlags                     = fread(fid, 1, 'uint32');
    header.samples                          = fread(fid, 1, 'uint32');
    header.portBeamWidthY                   = fread(fid, 1, 'float');
    header.portBeamWidthZ                   = fread(fid, 1, 'float');
    header.starboardBeamWidthY              = fread(fid, 1, 'float');
    header.starboardBeamWidthZ              = fread(fid, 1, 'float');
    header.portBeamSteeringAngleY           = fread(fid, 1, 'float');
    header.portBeamSteeringAngleZ           = fread(fid, 1, 'float');
    header.starboardBeamSteeringAngleY      = fread( fid, 1, 'float');
    header.starboardBeamSteeringAngleZ      = fread( fid, 1, 'float');
    header.noBeams                          = fread(fid, 1, 'uint16');
    header.currentBeamNumber                = fread(fid, 1, 'uint16');
    header.bytesPerSample                   = fread(fid, 1, 'uint8');
    header.dataTypes                        = fread(fid, 1, 'uint8');
    
    switch header.bytesPerSample
        case 1
            data.portBeams                  = fread(fid, header.samples * header.bytesPerSample, 'int8' );
            data.starboardBeams             = fread(fid, header.samples * header.bytesPerSample, 'int8' );
        case 2
            data.portBeams                  = fread(fid, header.samples * header.bytesPerSample / 2, 'int16' );
            data.starboardBeams             = fread(fid, header.samples * header.bytesPerSample / 2, 'int16' );
        case 4
            data.portBeams                  = fread(fid, header.samples * header.bytesPerSample / 4, 'int32' );
            data.starboardBeams             = fread(fid, header.samples * header.bytesPerSample / 4, 'int32' );
        otherwise
            warning( 'rsm:read7kRecordType', 'Unknown data format. Data read as unsigned 8 bit values' );
            data.portBeams                  = fread(fid, header.samples * header.bytesPerSample, 'uint8' );
            data.starboardBeams             = fread(fid, header.samples * header.bytesPerSample, 'uint8' );
    end
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes

    
elseif header.recordType == 7008 % water column data (obsolete)
    header.serialNumber                  = fread(fid, 1, 'uint64'); % 8 bytes,
    header.pingNumber              = fread(fid, 1, 'uint32'); % 4 bytes,
    header.multipingSequence       = fread(fid, 1, 'uint16'); % 2 bytes
    header.beams                   = fread(fid, 1, 'uint16'); % 2 bytes,
    header.reservedInt             = fread(fid, 1, 'uint16'); % 2 bytes,
    header.samples                 = fread(fid, 1, 'uint32'); % 4 bytes,
    header.subsetFlag              = fread(fid, 1, 'uint8');  % 1 byte,
    header.rowcolumnFlag           = fread(fid, 1, 'uint8');  % 1 byte,
    header.sampleHeaderIdentifier  = fread(fid, 1, 'uint16'); % 2 bytes,
    
%     header.sampleType              = fread(fid, 1, 'uint32');
    header.sampleType.magnitude    = fread(fid, 1, 'ubit4'); % bit 0-3,
    header.sampleType.phase        = fread(fid, 1, 'ubit4'); % bit 4-7,
    header.sampleType.IQ           = fread(fid, 1, 'ubit4'); % bit 8-11,
    header.sampleType.bfFlag       = fread(fid, 1, 'ubit3'); % bit 12-14,
    header.sampleType.notUsed      = fread(fid, 1, 'ubit17'); % bit 15-31,
     
    for n = 1:header.beams
        descriptor(n)              = fread(fid, 1, 'uint16'); 
        firstSampleNumber(n)       = fread(fid, 1, 'uint32'); 
        lastSampleNumber(n)        = fread(fid, 1, 'uint32'); 
    end
    
    switch header.sampleType.phase
        case 0
            NN = 1;
        case 2
            NN = 2;
    end
    
    header.sampleType = 0;
    
    if header.subsetFlag
        rawData                     = fread(fid, NN*length(firstSampleNumber(1):lastSampleNumber(1))*header.beams, 'uint16'); % 2 bytes
    else
        rawData                     = fread(fid, 2*header.samples*header.beams, 'uint16'); % 2 bytes
    end
    rawData(rawData<0) = rawData(rawData<0)+2^16;
    data.mag   = reshape( rawData( 1 : 2 : end ), [header.beams, header.samples] ).';
    data.phase = reshape( rawData( 2 : 2 : end ), [header.beams, header.samples] ).';
    data.phase(data.phase>=2^15) = data.phase(data.phase>=2^15)-2^16;

	header.checksum                 = fread(fid, 1, 'uint32');

elseif header.recordType == 7010
    header.serialNumber             = fread(fid, 1, 'uint64'); % 8 bytes,
    header.pingNumber               = fread(fid, 1, 'uint32'); % 4 bytes,
    header.multipingSequence        = fread(fid, 1, 'uint16'); % 2 bytes
    header.samples                  = fread(fid, 1, 'uint32'); % 4 bytes
    header.reserved                 = fread(fid, 8, 'uint32');
    
    for n = 1:header.samples
        data.gainValue(n) = fread(fid, 1, 'float'); % 4 bytes
    end
    
	header.checksum                 = fread(fid, 1, 'uint32');
    
elseif header.recordType == 7011
    header.pingNumber               = fread(fid, 1, 'uint32');
    header.multipingSequence        = fread(fid, 1, 'uint16'); % 2 bytes,14
    header.imageWidth               = fread(fid, 1, 'uint32');
    header.imageHeight              = fread(fid, 1, 'uint32');
    header.colorDepth               = fread(fid, 1, 'uint16');
    header.reservedFirst            = fread(fid, 1, 'uint16');
    header.compressionAlgorithm     = fread(fid, 1, 'uint16');
    header.samples                  = fread(fid, 1, 'uint32');
    header.reservedSecond           = fread(fid, 8, 'uint32');
    
%     usedWidth = max( header.imageWidth, 1024 );
%     usedHeight = max( header.imageHeight, 1024 );

    usedWidth = header.imageWidth;
    usedHeight = header.imageHeight;
    
    switch header.colorDepth
        case 1
            data.image                      = reshape( fread( fid, usedWidth * usedHeight * header.colorDepth, 'uint8' ), usedWidth, usedHeight );
        case 2
            data.image                      = reshape( fread( fid, usedWidth * usedHeight * header.colorDepth, 'uint16' ), usedWidth, usedHeight );
        case 4
            data.image                      = reshape( fread( fid, usedWidth * usedHeight * header.colorDepth, 'uint32' ), usedWidth, usedHeight );
        otherwise
%             warning( 'rsm:read7kRecordType', 'Undefined color depth, data read as 8 bit unsigned' );
            data.image                      = fread( fid, usedWidth * usedHeight * header.colorDepth, 'uint8' );
    end
    
    header.checksum                 = fread(fid, 1, 'uint32');

    
    
elseif header.recordType == 7012
   bytesRead = 64;
    
    header.sonarId                  = fread( fid, 1, 'uint64' ); bytesRead = bytesRead+8;
    header.pingNumber               = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+4;
    header.multipleSequence         = fread( fid, 1, 'uint16' ); bytesRead = bytesRead+2;
    header.samples                  = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+4;
    header.flags                    = fread( fid, 1, 'uint16' ); bytesRead = bytesRead+4;
    header.errorFlag                = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+2; % reading the error flag may cause problems on some files
    header.samplingRate             = fread( fid, 1, 'float32' ); bytesRead = bytesRead+4;
    bitField                        = ( dec2bin( header.flags, 16 ) == '1' );
%     if bitField(end) == true % !!!! CAUSED ERROR WITH 7160 DATA SET !!!!!!
%         header.pitch                = fread( fid, 1, 'float32' ); bytesRead = bytesRead+4;
%     else
%         header.pitch = [];
%     end;
%     if bitField(end-1) == true
%         header.roll                 = fread( fid, header.samples, 'float32' ); bytesRead = bytesRead+4*header.samples;
%     else
%         header.roll = [];
%     end;
%     if bitField(end-2) == true
%         header.heading              = fread( fid, header.samples, 'float32' ); bytesRead = bytesRead+4*header.samples;
%     else
%         header.heading = [];
%     end;
%     if bitField(end-3) == true
%         header.heave                = fread( fid, header.samples, 'float32' ); bytesRead = bytesRead+4*header.samples;
% 	else
%         header.heave = [];
%     end;
%     
%     header.checksum                 = fread(fid, 1, 'uint32'); bytesRead = bytesRead+4;
        
    bytesNotRead = header.size - bytesRead;
    if bytesNotRead>0
        fread(fid, bytesNotRead, 'uint8'); 
%         disp(['Record 7012: Read ' num2str(bytesNotRead) ' Extra bytes']);
    end

elseif header.recordType == 7017 % snippet data
    header.sonarId                              = fread(fid, 1, 'uint64');
    header.pingNumber                           = fread(fid, 1, 'uint32');
    header.multiPingSequence                    = fread(fid, 1, 'uint16');
    header.N                                    = fread(fid, 1, 'uint32');
    header.dataBlockSize                        = fread(fid, 1, 'uint32');
    header.detectionAlgorithm                   = fread(fid, 1, 'uint8');
    header.flags                                = fread(fid, 1, 'uint32');
    header.minimumDepth                         = fread(fid, 1, 'float');
    header.maximumDepth                         = fread(fid, 1, 'float');
    header.minimumRange                         = fread(fid, 1, 'float');
    header.maximumRange                         = fread(fid, 1, 'float');
    header.minimumNadirSearch                   = fread(fid, 1, 'float');
    header.maximumNadirSearch                   = fread(fid, 1, 'float');
    header.automaticFilterWindow                = fread(fid, 1, 'uint8');
    header.appliedRoll                          = fread(fid, 1, 'float');
    header.depthGateTilt                        = fread(fid, 1, 'float');
    header.nadirDepth                           = fread(fid, 1, 'float');
    header.reserved                             = fread(fid, 13, 'uint32');
    
    if header.size - 64 - 116 - 4 - header.N*34 > 0
        readUnknown7017 = 1;
    else
        readUnknown7017 = 0;
    end
    
    for n=1:header.N
        data.beamDescriptor(n)                      = fread(fid, 1, 'uint16');
        data.detectionPoint(n)                      = fread(fid, 1, 'float');
        data.flags(n)                               = fread(fid, 1, 'uint32');
        data.automaticLimitsMinDetectionSample(n)   = fread(fid, 1, 'float');
        data.automaticLimitsMaxDetectionSample(n)   = fread(fid, 1, 'float');
        data.userLimitsMinSample(n)                 = fread(fid, 1, 'float');
        data.userLimitsMaxSample(n)                 = fread(fid, 1, 'float');
        data.quality(n)                             = fread(fid, 1, 'bit32');
        data.uncertainty(n)                         = fread(fid, 1, 'float');
        if readUnknown7017
            data.unknown(n)                             = fread(fid, 1, 'float');
        end
    end
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes

    
elseif header.recordType == 7018
    header.sonarId                              = fread(fid, 1, 'uint64'); % 8 bytes
    header.pingNumber                           = fread(fid, 1, 'uint32'); % 4 bytes
    header.multipingSequence                    = fread(fid, 1, 'uint16'); % 2 bytes
    header.beams                                = fread(fid, 1, 'uint16'); % 2 bytes
    header.samples                              = fread(fid, 1, 'uint32'); % 4 bytes
    header.reserved                             = fread(fid, 8, 'uint32'); % 32 bytes
    
    rawData                                     = fread(fid, 2*header.samples*header.beams, 'uint16'); % 2 bytes
    
    data.mag  = reshape( rawData( 1 : 2 : end ), [header.beams, header.samples] ).';
    data.phase      = reshape( rawData( 2 : 2 : end ), [header.beams, header.samples] ).';
    data.phase(data.phase>=2^15) = data.phase(data.phase>=2^15)-2^16;

    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
    
elseif header.recordType == 7021
    header.noBoards = fread(fid, 1, 'uint16');
    for b = 1:header.noBoards
        data.sourceName{b} = fread(fid, 64, 'uint8');
        data.sourceAddress{b} = fread(fid, 1, 'uint8');
        data.frequency{b} = fread(fid, 1, 'float');
        data.enumerator{b} = fread(fid, 1, 'uint16');
        data.downlinkTimeSent{b} = fread(fid, 10, 'uint8');
        data.uplinkTimeReceived{b} = fread(fid, 10, 'uint8');
        data.biteTimeReceived{b} = fread(fid, 10, 'uint8');
        data.status{b} = fread(fid, 1, 'uint8');
        data.nbf{b} = fread(fid, 1, 'uint16');
        data.biteStatusBit{b} = fread(fid, 4, 'uint64');
        for n = 1:data.nbf{b}
            data.bfField{b,n} = fread(fid, 1, 'uint16');
            data.bfName{b,n} = char(fread(fid, 64, 'char')');
            data.bfSensorType{b,n} = fread(fid, 1, 'uint8');
            data.bfMinimum{b,n} = fread(fid, 1, 'float');
            data.bfMaximum{b,n} = fread(fid, 1, 'float');
            data.bfValue{b,n} = fread(fid, 1, 'float');
        end
    end
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7022
    header.version7kC                           = fread(fid, 32, 'uint8'); % 8 bytes
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
    
elseif header.recordType == 7027
    header.sonarId                              = fread(fid, 1, 'uint64'); % 8 bytes
    header.pingNumber                           = fread(fid, 1, 'uint32'); % 4 bytes
    header.multiPingSequence                    = fread(fid, 1, 'uint16'); % 2 bytes
    header.N                                    = fread(fid, 1, 'uint32'); % 4 bytes
    header.dataFieldSize                        = fread(fid, 1, 'uint32'); % 4 bytes
    header.detectionAlgorithm                   = fread(fid, 1, 'uint8'); % 4 bytes
    header.flags                                = fread(fid, 1, 'uint32'); % 4 bytes
    header.samplingRate                         = fread(fid, 1, 'float32'); % 4 bytes
    header.txAngle                              = fread(fid, 1, 'float32'); % 4 bytes
    header.reserved                             = fread(fid, 16, 'uint32'); % 64 bytes
    
    if header.N==0
        data=[];
    end
    
    for n=1:header.N
        data.beamDescriptor(n)                  = fread(fid, 1, 'uint16');
        data.detectionPoint(n)                  = fread(fid, 1, 'float32');
        data.rxAngle(n)                         = fread(fid, 1, 'float32');
        data.flags(n)                           = fread(fid, 1, 'uint32');
        data.quality(n)                         = fread(fid, 1, 'uint32');
        data.uncertainty(n)                     = fread(fid, 1, 'float32');
        if header.dataFieldSize-22 > 0
            data.reserved                       = fread(fid, header.dataFieldSize-22, 'uint8');
        end
    end
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 7028 % 7k Snippet data
    header.sonarId                              = fread( fid, 1, 'uint64');
    header.pingNumber                           = fread( fid, 1, 'uint32');
    header.multiPingSequence                    = fread( fid, 1, 'uint16');
    header.detections                           = fread( fid, 1, 'uint16' );
    header.errorFlag                            = fread( fid, 1, 'uint8');
    header.controlFlag                          = fread( fid, 1, 'uint8' );
    header.reserved                             = fread( fid, 7, 'uint32' );
    
    for n=1:header.detections
        data.beamNumber(n)                      = fread( fid, 1, 'uint16' );
        data.beginSample(n)                     = fread( fid, 1, 'uint32' );
        data.detectionSample(n)                 = fread( fid, 1, 'uint32' );
        data.endSample(n)                       = fread( fid, 1, 'uint32' );
    end
    
    for n=1:header.detections
        data.snippet{n} = fread(fid, data.endSample(n)-data.beginSample(n) + 1, 'uint16');
    end
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7032 % 7k Snippet data
    header.sonarId                              = fread( fid, 1, 'uint64');
    header.pingNumber                           = fread( fid, 1, 'uint32');
    header.nPulses                              = fread( fid, 1, 'uint8');
    header.startFrequency                       = fread( fid, 1, 'float');
    header.stopFrequency                        = fread( fid, 1, 'float');
    header.outputPulseLength                    = fread( fid, 1, 'float');
    header.interPulseGap                        = fread( fid, 1, 'float');
    header.reserved                             = fread( fid, 4, 'uint32');
    
    header.checksum                             = fread( fid, 1, 'uint32');
        
    
elseif header.recordType == 7038
%     header.serialNumber                         = fread(fid, 1, 'uint64');
% 	header.pingNumber                           = fread(fid, 1, 'uint32');
% 	header.multipingSequence                    = fread(fid, 1, 'uint16');
% 	header.channelCount                         = fread(fid, 1, 'uint16');
% 	header.noSamples                            = fread(fid, 1, 'uint32');
% 	header.actualChannels                       = fread(fid, 1, 'uint16');
% 	header.startSample                          = fread(fid, 1, 'uint32');
% 	header.stopSample                           = fread(fid, 1, 'uint32');
% 	header.sampleType                           = fread(fid, 1, 'uint16');
% 	header.reserved                             = fread(fid, 1, 'uint32');
%     
%     actualSamples = header.stopSample-header.startSample+1;
%     
%     temp                                        = fread(fid, header.actualChannels*header.noSamples *2, 'integer*2');
%     
%     data.I  = reshape( temp(1:2:end), [header.actualChannels, header.noSamples] ).'
%     data.Q  = reshape( temp(2:2:end), [header.actualChannels, header.noSamples] ).'
    header.serialNumber                         = fread(fid, 1, 'uint64');
	header.pingNumber                           = fread(fid, 1, 'uint32');
	header.multipingSequence                    = fread(fid, 1, 'uint16');
	header.channelCount                         = fread(fid, 1, 'uint16');
	header.noSamples                            = fread(fid, 1, 'uint32');
	header.actualChannels                       = fread(fid, 1, 'uint16');
	header.startSample                          = fread(fid, 1, 'uint32');
	header.stopSample                           = fread(fid, 1, 'uint32');
	header.sampleType                           = fread(fid, 1, 'uint16');
	header.reserved                             = fread(fid, 7, 'uint32');
    
%     actualSamples = header.stopSample-header.startSample+1;
%     
%     temp                                        = fread(fid, header.actualChannels*header.noSamples *2, 'integer*2');
%     
%     data.I  = reshape( temp(1:2:end), [header.actualChannels, header.noSamples] ).'
%     data.Q  = reshape( temp(2:2:end), [header.actualChannels, header.noSamples] ).'

    %
    header.channelArray = fread( fid, header.actualChannels, 'uint16' );
    actualSamples = header.stopSample - header.startSample + 1;
    data.value = zeros( header.actualChannels, header.noSamples );
    
    if header.sampleType == 8
        actualData = fread( fid, header.actualChannels * actualSamples * 2, 'int8' );
        actualData( actualData < 0 ) = 256 + actualData( actualData < 0 );
        actualData = actualData * 16;
        actualData( actualData > 2047 ) = actualData( actualData > 2047 ) - 4096;
    elseif header.sampleType == 16
        actualData = fread( fid, header.actualChannels * actualSamples * 2, 'int16' );
%         actualData( actualData < 0 ) = 65536 + actualData( actualData < 0 );
%         actualData = actualData / 16;
%         actualData( actualData > 2047 ) = actualData( actualData > 2047 ) - 4096;
    else
        actualData = fread( fid, header.actualChannels * actualSamples * 2, 'int16');
    end
    
    iData = actualData(1:2:end);
    qData = actualData(2:2:end);
    iData = reshape( iData, header.actualChannels, [] );
    qData = reshape( qData, header.actualChannels, [] );
    data.value( header.channelArray + 1, header.startSample + 1 : header.stopSample + 1 ) = iData + 1i * qData;
    data.value = data.value.';
    
    header.channelArray = header.channelArray + 1;
    %
    header.checksum           = fread(fid, 1, 'uint32');

    
elseif header.recordType == 7200
    bytesRead = 64;
    
    header.fileId                               = fread( fid, 16, 'uint8' ); bytesRead = bytesRead+16;
    header.versionNumber                        = fread( fid, 1, 'uint16' ); bytesRead = bytesRead+2;
    header.reserved                             = fread( fid, 1, 'uint16' ); bytesRead = bytesRead+2;
    header.sessionId                            = fread( fid, 16, 'uint8' ); bytesRead = bytesRead+16;
    header.recordDataSize                       = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+4;
    header.noDevices                            = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+4;
    header.recordingName                        = char( fread( fid, 64, 'char' ))'; bytesRead = bytesRead+64;
    header.recordingProgramVersion              = char( fread( fid, 16, 'char' ))'; bytesRead = bytesRead+16;
    header.userName                             = char( fread( fid, 64, 'char' ))'; bytesRead = bytesRead+64;
    header.notes                                = char( fread( fid, 128, 'char' ))'; bytesRead = bytesRead+128;
    for nDevice = 1 : header.noDevices % +1
        data.id(nDevice)                        = fread( fid, 1, 'uint32' ); bytesRead = bytesRead+4;
        data.systemEnumerator(nDevice)          = fread( fid, 1, 'uint16' ); bytesRead = bytesRead+2;
    end
    
  	header.checksum           = fread(fid, 1, 'uint32'); bytesRead = bytesRead+4;

    bytesNotRead = header.size - bytesRead;
    if bytesNotRead
        fread(fid, bytesNotRead, 'uint8'); 
%         disp(['Record 7200: Read ' num2str(bytesNotRead) ' Extra bytes']);
    end
   
    
elseif header.recordType == 7300
    header.size                     = fread( fid, 1, 'uint32' );
    header.version                  = fread( fid, 1, 'uint16' );
    header.numberOfRecords          = fread( fid, 1, 'uint32' );
    header.reserved                 = fread( fid, 1, 'uint32' );
    
    for n = 1:header.numberOfRecords
        data.size(n)                = fread( fid, 1, 'uint32' );
        data.offset(n)              = fread( fid, 1, 'uint64' );
        data.recordType(n)          = fread( fid, 1, 'uint16' );
        data.deviceIdentifier(n)    = fread( fid, 1, 'uint16' );
        data.systemEnnumerator(n)   = fread( fid, 1, 'uint16' );
        data.time7k{n}              = fread( fid, 10, 'uint8' );
        data.recordCount(n)         = fread( fid, 1, 'uint32' );
        data.reserved{n}            = fread( fid, 8, 'uint16' );
    end
    
  	header.checksum                 = fread( fid, 1, 'uint32' );

    
elseif header.recordType == 7503
    header.sonarId = fread( fid, 1, 'uint64' );
    header.pingNumber = fread( fid, 1, 'uint32' );
    header.frequency = fread( fid, 1, 'single' );
    header.sampleRate = fread( fid, 1, 'single' );
    header.receiverBandwidth = fread( fid, 1, 'single' );
    header.txPulseWidth = fread( fid, 1, 'single' );
    header.txPulseType = fread( fid, 1, 'uint32' );
    header.txPulseEnvelope = fread( fid, 1, 'uint32' );
    header.txPulseEnvelopeParameter = fread( fid, 1, 'single' );
    header.txPulse = fread( fid, 1, 'uint32' );
    header.maxPingRate = fread( fid, 1, 'single' );
    header.pingPeriod = fread( fid, 1, 'single' );
    header.range = fread( fid, 1, 'single' );
    header.power = fread( fid, 1, 'single' );
    header.gain = fread( fid, 1, 'single' );
    header.controlFlags = fread( fid, 1, 'uint32' );
    header.projectorId = fread( fid, 1, 'uint32' );
    header.projectorSteeringAngleVertical = fread( fid, 1, 'single' );
    header.projectorSteeringAngleHorizontal = fread( fid, 1, 'single' );
    header.projectorBeamWidthVertical = fread( fid, 1, 'single' );
    header.projectorBeamWidthHorizontal = fread( fid, 1, 'single' );
    header.projectorFocalPoint = fread( fid, 1, 'single' );
    header.projectorBeamWindow = fread( fid, 1, 'uint32' );
    header.projectorBeamParam = fread( fid, 1, 'single' );
    header.transmitFlags = fread( fid, 1, 'uint32' );
    header.hydrophoneId = fread( fid, 1, 'uint32' );
    header.receiveBeamWindow = fread( fid, 1, 'uint32' );
    header.receiveBeamParam = fread( fid, 1, 'single' );
    header.receiveFlags = fread( fid, 1, 'uint32' );
    header.bottomDetectionMinRange = fread( fid, 1, 'single' );
    header.bottomDetectionMaxRange = fread( fid, 1, 'single' );
    header.bottomDetectionMinDepth = fread( fid, 1, 'single' );
    header.bottomDetectionMaxDepth = fread( fid, 1, 'single' );
    header.absorption = fread( fid, 1, 'single' );
    header.soundVelocity = fread( fid, 1, 'single' );
    header.spreading = fread( fid, 1, 'single' );
    header.reservedInt = fread( fid, 1, 'uint16' );
    header.txArrayPositionX = fread( fid, 1, 'single' );
    header.txArrayPositionY = fread( fid, 1, 'single' );
    header.txArrayPositionZ = fread( fid, 1, 'single' );
    header.headTiltX = fread( fid, 1, 'single' );
    header.headTiltY = fread( fid, 1, 'single' );
    header.headTiltZ = fread( fid, 1, 'single' );
    header.pingState = fread( fid, 1, 'uint32' );
    header.angularMode = fread( fid, 1, 'uint16' );
    header.centerMode = fread( fid, 1, 'uint16' );
    header.adaptiveGateMinDepth = fread( fid, 1, 'single' );
    header.adaptiveGateMaxDepth = fread( fid, 1, 'single' );
    header.triggerOutWidth = fread( fid, 1, 'double' );
    header.triggerOutOffset = fread( fid, 1, 'double' );
    header.projector8kSeries = fread( fid, 1, 'uint16' );
    header.reservedSpecific = fread( fid, 2, 'uint32' );
    header.alternateGain8kSeries = fread( fid, 1, 'single' );
    header.reservedWord = fread( fid, 1, 'uint32' );
    header.coverageAngle = fread( fid, 1, 'single' );
    header.coverageMode = fread( fid, 1, 'uint8' );
    header.qualityFilterFlags = fread( fid, 1, 'uint8' );
    header.reservedCharArray = fread( fid, 2, 'uint8' );
    header.reservedWordArray = fread( fid, 7, 'uint32' );

    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7504
    header.serialId                 = fread( fid, 1, 'uint64' );
    header.pingNumber               = fread( fid, 1, 'uint32' );
    header.soundVelocity            = fread( fid, 1, 'float' );
    header.absorption               = fread( fid, 1, 'float' );
    header.spreadingLoss            = fread( fid, 1, 'float' );
    header.sequencerControl         = fread( fid, 1, 'uint32' );
    header.motionSensorFormat       = fread( fid, 1, 'uint8' );
    header.motionSensorBaudRate     = fread( fid, 1, 'uint8' );
    header.motionSensorParity       = fread( fid, 1, 'uint8' );
    header.motionSensorDataBits     = fread( fid, 1, 'uint8' );
    header.motionSensorStopBits     = fread( fid, 1, 'uint8' );
    header.orientation              = fread( fid, 1, 'uint8' );
    header.orientationInverted      = fread( fid, 1, 'uint8' );
    header.motionLatency            = fread( fid, 1, 'float' );
    header.reserved                 = fread( fid, 1, 'uint8' );
    header.manualOverride           = fread( fid, 1, 'uint8' );
    header.activeEnumetator         = fread( fid, 1, 'uint16' );
    header.activeDeviceId           = fread( fid, 1, 'uint32' );
    header.systemMode               = fread( fid, 1, 'uint32' );
    header.reserved2                = fread( fid, 123, 'uint32' );
    
    header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 7510
    
	header.sensorSV                             = fread(fid, 1, 'float32'); % 2 bytes
    header.filteredSV                           = fread(fid, 1, 'float32'); % 2 bytes
    header.filter                               = fread(fid, 1, 'uint8'); % 2 bytes
    
    header.checksum                             = fread(fid, 1, 'uint32'); % 2 bytes
        
elseif header.recordType == 7610
    bytesRead = 64;
    bytesNotRead = header.size - bytesRead;
    
    if bytesNotRead == 8
        header.soundVelocity                        = fread(fid, 1, 'float32'); bytesRead = bytesRead +4;
    elseif bytesNotRead > 8
        header.soundVelocity                        = fread(fid, 1, 'float32'); bytesRead = bytesRead +4;
        header.temperature                          = fread(fid, 1, 'float32'); bytesRead = bytesRead +4;
        header.pressure                             = fread(fid, 1, 'float32'); bytesRead = bytesRead +4;
    end
    header.checksum                             = fread(fid, 1, 'uint32'); bytesRead = bytesRead +4;
% THE BELOW LINES FIXED PROBLEMS READING SVP DATA, THEN UPDATED WITH TEMP
% AND PRESSURE. THEN TRIED WITH 'IF-ELSEIF'
%     bytesNotRead = header.size - bytesRead;
%     if bytesNotRead
%         fread(fid, bytesNotRead, 'uint8'); 
%         disp(['Record 7610: Read ' num2str(bytesNotRead) ' Extra bytes']);
%     end

elseif header.recordType == 10000

    if isNewFormat
        header.pingNumber               = fread( fid, 1, 'uint32' );
        header.multiPingSequence        = fread( fid, 1, 'uint32' );
        header.timeState                = fread( fid, 1, 'uint32' );
        header.transportLatency         = fread( fid, 1, 'uint32' );
        header.centerFrequency          = fread( fid, 1, 'float' );
        header.sweepWidth               = fread( fid, 1, 'float' );
        header.sampleRate               = fread( fid, 1, 'float' );
        header.receiverBandwidth        = fread( fid, 1, 'float' );
        header.absorption               = fread( fid, 1, 'float' );
        header.spreading                = fread( fid, 1, 'float' );
        header.gain                     = fread( fid, 1, 'float' );
        header.range                    = fread( fid, 1, 'float' );
        header.barCheckOffset           = fread( fid, 1, 'float' );
        header.power                    = fread( fid, 1, 'float' );
        header.pulseWidth               = fread( fid, 1, 'float' );
        header.pulseType                = fread( fid, 1, 'uint32' );
        header.pulseEnvelopeType        = fread( fid, 1, 'uint32' );
        header.pulseEnvelopeParameter	= fread( fid, 1, 'float' );
        header.multipingCount           = fread( fid, 1, 'uint32' );
        header.reserved                 = fread( fid, 1, 'uint32' );
        header.maxPingRate              = fread( fid, 1, 'float' );
        header.pingPeriod               = fread( fid, 1, 'float' );
        header.rxMinGate                = fread( fid, 1, 'float' );
        header.projectorId              = fread( fid, 1, 'uint32' );
        header.txBeamWidthAcross        = fread( fid, 1, 'float' );
        header.txBeamWidthAlong         = fread( fid, 1, 'float' );
        header.receiverId               = fread( fid, 1, 'uint32' );
        header.rxBeamWidthAcross        = fread( fid, 1, 'float' );
        header.rxBeamWidthAlong         = fread( fid, 1, 'float' );
        header.correctedTransducerDepth = fread( fid, 1, 'float' );
        header.tweakAbsorption          = fread( fid, 1, 'float' );
        header.tweakPulseLength         = fread( fid, 1, 'float' );
        header.tweakSpreadingLoss       = fread( fid, 1, 'float' );
        header.tweakInitGain            = fread( fid, 1, 'float' );
        header.tweakRange               = fread( fid, 1, 'float' );
        header.tweakPower               = fread( fid, 1, 'float' );
    else % old format
        header.pingNumber               = fread( fid, 1, 'uint32' );
        header.multiPingSequence        = fread( fid, 1, 'uint32' );
        header.timeState                = fread( fid, 1, 'uint32' );
        header.transportLatency         = fread( fid, 1, 'uint32' );
        header.centerFrequency          = fread( fid, 1, 'float' );
        header.sweepWidth               = fread( fid, 1, 'float' );
        header.sampleRate               = fread( fid, 1, 'float' );
        header.receiverBandwidth        = fread( fid, 1, 'float' );
        header.absorption               = fread( fid, 1, 'float' );
        header.spreading                = fread( fid, 1, 'float' );
        header.gain                     = fread( fid, 1, 'float' );
        header.soundVelocity            = fread( fid, 1, 'float' ); %% SEE NOTE
        header.range                    = fread( fid, 1, 'float' );
%         header.barCheckOffset           = fread( fid, 1, 'float' );
        header.power                    = fread( fid, 1, 'float' );
        header.pulseWidth               = fread( fid, 1, 'float' );
        header.pulseType                = fread( fid, 1, 'uint32' );
        header.pulseEnvelopeType        = fread( fid, 1, 'uint32' );
        header.pulseEnvelopeParameter	= fread( fid, 1, 'float' );
        header.multipingCount           = fread( fid, 1, 'uint32' );
        header.reserved                 = fread( fid, 1, 'uint32' );
        header.maxPingRate              = fread( fid, 1, 'float' );
        header.pingPeriod               = fread( fid, 1, 'float' );
        header.rxMinGate                = fread( fid, 1, 'float' );
        header.projectorId              = fread( fid, 1, 'uint32' );
        header.txBeamWidthAcross        = fread( fid, 1, 'float' );
        header.txBeamWidthAlong         = fread( fid, 1, 'float' );
        header.receiverId               = fread( fid, 1, 'uint32' );
        header.rxBeamWidthAcross        = fread( fid, 1, 'float' );
        header.rxBeamWidthAlong         = fread( fid, 1, 'float' );
        header.correctedTransducerDepth = fread( fid, 1, 'float' );
        header.averageSoundVelocity     = fread( fid, 1, 'float' );
    end
    
	header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes

elseif header.recordType == 10018
    if isNewFormat
        header.pingNumber               = fread( fid, 1, 'uint32' );
        header.multiPingSequence        = fread( fid, 1, 'uint32' );
        header.meanSoundVelocity        = fread( fid, 1, 'float' );
        header.correctedTransducerDepth = fread( fid, 1, 'float' );
        header.sampleRate               = fread( fid, 1, 'float' );
        header.txPulseShift             = fread( fid, 1, 'float' );
        header.startSampleDelay         = fread( fid, 1, 'float' );
        header.bitsPrSample             = fread( fid, 1, 'uint32' );
        header.fullScale                = fread( fid, 1, 'uint32' );
        header.nSamples                 = fread( fid, 1, 'uint32' );
%         header.reserved                 = fread( fid, 20, 'uint32' );
        header.sampleData               = uint32(fread( fid, header.nSamples, 'uint32' ));
    else
        header.pingNumber               = fread( fid, 1, 'uint32' );
        header.multiPingSequence        = fread( fid, 1, 'uint32' );
        header.meanSoundVelocity        = fread( fid, 1, 'float' );
        header.correctedTransducerDepth = fread( fid, 1, 'float' );
        header.sampleRate               = fread( fid, 1, 'float' );
        header.txPulseShift             = fread( fid, 1, 'float' );
        header.startSampleDelay         = fread( fid, 1, 'float' );
        header.bitsPrSample             = fread( fid, 1, 'uint32' );
        header.fullScale                = fread( fid, 1, 'uint32' );
        header.nSamples                 = fread( fid, 1, 'uint32' );
        header.sampleData               = uint32(fread( fid, header.nSamples, 'uint32' ));
    end
    header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes
    
elseif header.recordType == 10027
    header.pingNumber               = fread( fid, 1, 'uint32' );
    header.multiPingSequence        = fread( fid, 1, 'uint32' );
    header.detectionCount           = fread( fid, 1, 'uint32' );
    header.dataFieldSize            = fread( fid, 1, 'uint32' );
    header.meanSoundVelocity        = fread( fid, 1, 'float' );
    header.sampleRate               = fread( fid, 1, 'float' );
    header.reserved                 = fread( fid, 4, 'float' );
    header.notAppliedCorrTcDepth    = fread( fid, 1, 'float' );
    
    header.detectionMethod          = zeros(1,1);
    header.range                    = zeros(1,1);
    header.notAppliedHeave          = zeros(1,1);
    header.amplitude                = zeros(1,1);
    header.snr                      = zeros(1,1);
    header.seabedConfidence         = -999*ones(1,1);
    header.colinearity              = zeros(1,1);
    header.classifier               = zeros(1,1);
    for n = 1:header.detectionCount
        header.detectionMethod(n)   = fread( fid, 1, 'uint32' );
        header.range(n)             = fread( fid, 1, 'float' );
        header.notAppliedHeave(n)   = fread( fid, 1, 'float' );
        header.amplitude(n)         = fread( fid, 1, 'float' );
        header.snr(n)               = fread( fid, 1, 'float' );
        header.seabedConfidence(n)  = fread( fid, 1, 'float' );
        header.colinearity(n)       = fread( fid, 1, 'float' );
        header.classifier(n)        = fread( fid, 1, 'uint32' );
    end
    header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes
    

elseif header.recordType == 10038
    header.pingNumber               = fread( fid, 1, 'uint32' );
    header.MultiPingSequence        = fread( fid, 1, 'uint32' );
    header.meanSoundVelocity        = fread( fid, 1, 'float' );
    header.notAppliedTcDraft        = fread( fid, 1, 'float' );
    header.sampleRate               = fread( fid, 1, 'float' );
    header.effectivePulseLength     = fread( fid, 1, 'float' );
    header.startSampleDelay         = fread( fid, 1, 'float' );
    header.bitsPerSample            = fread( fid, 1, 'uint32' );
    header.fullScale                = fread( fid, 1, 'float' );
    header.nSamples                 = fread( fid, 1, 'uint32' );
    header.reserved                 = fread( fid, 20, 'uint32' );
    
    dataRaw                         = fread( fid, 2*header.nSamples, 'float' );
    header.sampleDataRe = dataRaw(1:2:end);
    header.sampleDataIm = dataRaw(2:2:end);
        
    header.checksum                 = fread(fid, 1, 'uint32'); % 2 bytes
    
else
    if ~isempty(header.size)
        fseek(fid, header.size-64, 0);
    end
end