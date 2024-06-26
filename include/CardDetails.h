#ifndef CARD_DETAILS_H
#define CARD_DETAILS_H

#include <string>
#include <unordered_map>

namespace CardDetails
{
enum FrameColor
{
    Unsure,
    Colorless,
    Red,
    Blue,
    Green,
    White,
    Black,
    Multi,
    Land_Color
};

enum Type
{
    Unidentified,
    Instant,
    Sorcery,
    Artifact,
    Creature,
    Enchantment,
    Land,
    Planeswalker
};

static const std::unordered_map<std::string, Type> Type_parse({{"Unidentified", Unidentified},
                                                               {"Instant", Instant},
                                                               {"Sorcery", Sorcery},
                                                               {"Artifact", Artifact},
                                                               {"Creature", Creature},
                                                               {"Enchantment", Enchantment},
                                                               {"Land", Land},
                                                               {"Planeswalker", Planeswalker}});

enum CardSet
{
    Unknown,
    _10E,
    _2ED,
    _3ED,
    _4ED,
    _5DN,
    _5ED,
    _6ED,
    _7ED,
    _8ED,
    _9ED,
    ALA,
    ALL,
    APC,
    ARB,
    ARC,
    ARN,
    ATH,
    ATQ,
    AVR,
    BNG,
    BOK,
    BRB,
    BTD,
    C13,
    C14,
    CHK,
    CHR,
    CM1,
    CMD,
    CNS,
    CSP,
    CST,
    DD2,
    DD3_DVD,
    DD3_EVG,
    DD3_GVL,
    DD3_JVC,
    DDC,
    DDD,
    DDE,
    DDF,
    DDG,
    DDH,
    DDI,
    DDJ,
    DDK,
    DDL,
    DDM,
    DDN,
    DGM,
    DIS,
    DKA,
    DKM,
    DPA,
    DRB,
    DRK,
    DST,
    EVE,
    EVG,
    EXO,
    FEM,
    FRF,
    FUT,
    GPT,
    GTC,
    H09,
    HML,
    HOP,
    ICE,
    INV,
    ISD,
    ITP,
    JOU,
    JUD,
    KTK,
    LEA,
    LEB,
    LEG,
    LGN,
    LRW,
    M10,
    M11,
    M12,
    M13,
    M14,
    M15,
    MBS,
    MD1,
    ME2,
    ME3,
    ME4,
    MED,
    MGB,
    MIR,
    MMA,
    MMQ,
    MOR,
    MRD,
    NMS,
    NPH,
    ODY,
    ONS,
    p15A,
    p2HG,
    pALP,
    pARL,
    PC2,
    pCEL,
    pCMP,
    PCY,
    PD2,
    PD3,
    pDRC,
    pELP,
    pFNM,
    pGPX,
    pGRU,
    pHHO,
    pJGP,
    PLC,
    pLGM,
    pLPA,
    PLS,
    pMEI,
    pMGD,
    pMPR,
    PO2,
    POR,
    pPOD,
    pPRE,
    pPRO,
    pREL,
    pSUM,
    pSUS,
    PTK,
    pWCQ,
    pWOR,
    pWOS,
    pWPN,
    RAV,
    ROE,
    RQS,
    RTR,
    S00,
    S99,
    SCG,
    SHM,
    SOK,
    SOM,
    STH,
    THS,
    TMP,
    TOR,
    TSB,
    TSP,
    UDS,
    UGL,
    ULG,
    UNH,
    USG,
    V09,
    V10,
    V11,
    V12,
    V13,
    V14,
    VAN,
    VIS,
    VMA,
    WTH,
    WWK,
    ZEN
};
static const std::unordered_map<CardSet, std::string> CardSet_name(
    {{_10E, "10E"},        {_2ED, "2ED"},        {_3ED, "3ED"},  {_4ED, "4ED"},        {_5DN, "5DN"},
     {_5ED, "5ED"},        {_6ED, "6ED"},        {_7ED, "7ED"},  {_8ED, "8ED"},        {_9ED, "9ED"},
     {ALA, "ALA"},         {ALL, "ALL"},         {APC, "APC"},   {ARB, "ARB"},         {ARC, "ARC"},
     {ARN, "ARN"},         {ATH, "ATH"},         {ATQ, "ATQ"},   {AVR, "AVR"},         {BNG, "BNG"},
     {BOK, "BOK"},         {BRB, "BRB"},         {BTD, "BTD"},   {C13, "C13"},         {C14, "C14"},
     {CHK, "CHK"},         {CHR, "CHR"},         {CM1, "CM1"},   {CMD, "CMD"},         {CNS, "CNS"},
     {CSP, "CSP"},         {CST, "CST"},         {DD2, "DD2"},   {DD3_DVD, "DD3_DVD"}, {DD3_EVG, "DD3_EVG"},
     {DD3_GVL, "DD3_GVL"}, {DD3_JVC, "DD3_JVC"}, {DDC, "DDC"},   {DDD, "DDD"},         {DDE, "DDE"},
     {DDF, "DDF"},         {DDG, "DDG"},         {DDH, "DDH"},   {DDI, "DDI"},         {DDJ, "DDJ"},
     {DDK, "DDK"},         {DDL, "DDL"},         {DDM, "DDM"},   {DDN, "DDN"},         {DGM, "DGM"},
     {DIS, "DIS"},         {DKA, "DKA"},         {DKM, "DKM"},   {DPA, "DPA"},         {DRB, "DRB"},
     {DRK, "DRK"},         {DST, "DST"},         {EVE, "EVE"},   {EVG, "EVG"},         {EXO, "EXO"},
     {FEM, "FEM"},         {FRF, "FRF"},         {FUT, "FUT"},   {GPT, "GPT"},         {GTC, "GTC"},
     {H09, "H09"},         {HML, "HML"},         {HOP, "HOP"},   {ICE, "ICE"},         {INV, "INV"},
     {ISD, "ISD"},         {ITP, "ITP"},         {JOU, "JOU"},   {JUD, "JUD"},         {KTK, "KTK"},
     {LEA, "LEA"},         {LEB, "LEB"},         {LEG, "LEG"},   {LGN, "LGN"},         {LRW, "LRW"},
     {M10, "M10"},         {M11, "M11"},         {M12, "M12"},   {M13, "M13"},         {M14, "M14"},
     {M15, "M15"},         {MBS, "MBS"},         {MD1, "MD1"},   {ME2, "ME2"},         {ME3, "ME3"},
     {ME4, "ME4"},         {MED, "MED"},         {MGB, "MGB"},   {MIR, "MIR"},         {MMA, "MMA"},
     {MMQ, "MMQ"},         {MOR, "MOR"},         {MRD, "MRD"},   {NMS, "NMS"},         {NPH, "NPH"},
     {ODY, "ODY"},         {ONS, "ONS"},         {p15A, "p15A"}, {p2HG, "p2HG"},       {pALP, "pALP"},
     {pARL, "pARL"},       {PC2, "PC2"},         {pCEL, "pCEL"}, {pCMP, "pCMP"},       {PCY, "PCY"},
     {PD2, "PD2"},         {PD3, "PD3"},         {pDRC, "pDRC"}, {pELP, "pELP"},       {pFNM, "pFNM"},
     {pGPX, "pGPX"},       {pGRU, "pGRU"},       {pHHO, "pHHO"}, {pJGP, "pJGP"},       {PLC, "PLC"},
     {pLGM, "pLGM"},       {pLPA, "pLPA"},       {PLS, "PLS"},   {pMEI, "pMEI"},       {pMGD, "pMGD"},
     {pMPR, "pMPR"},       {PO2, "PO2"},         {POR, "POR"},   {pPOD, "pPOD"},       {pPRE, "pPRE"},
     {pPRO, "pPRO"},       {pREL, "pREL"},       {pSUM, "pSUM"}, {pSUS, "pSUS"},       {PTK, "PTK"},
     {pWCQ, "pWCQ"},       {pWOR, "pWOR"},       {pWOS, "pWOS"}, {pWPN, "pWPN"},       {RAV, "RAV"},
     {ROE, "ROE"},         {RQS, "RQS"},         {RTR, "RTR"},   {S00, "S00"},         {S99, "S99"},
     {SCG, "SCG"},         {SHM, "SHM"},         {SOK, "SOK"},   {SOM, "SOM"},         {STH, "STH"},
     {THS, "THS"},         {TMP, "TMP"},         {TOR, "TOR"},   {TSB, "TSB"},         {TSP, "TSP"},
     {UDS, "UDS"},         {UGL, "UGL"},         {ULG, "ULG"},   {UNH, "UNH"},         {USG, "USG"},
     {V09, "V09"},         {V10, "V10"},         {V11, "V11"},   {V12, "V12"},         {V13, "V13"},
     {V14, "V14"},         {VAN, "VAN"},         {VIS, "VIS"},   {VMA, "VMA"},         {WTH, "WTH"},
     {WWK, "WWK"},         {ZEN, "ZEN"}});
} // namespace CardDetails

#endif // CARD_DETAILS_H
