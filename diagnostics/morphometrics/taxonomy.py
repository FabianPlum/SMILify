"""Genus -> higher-taxon labels, used only as GROUPING LABELS for the morphometric analysis.

Nothing here is derived from the scans: it is external ground truth, which is the point. A
clustering result is only meaningful against labels that were fixed independently of the data
being clustered.

Classification follows AntCat/AntWiki, cross-checked against the primary revisions that define
the modern tribes -- Ward, Brady, Fisher & Schultz 2015 (Myrmicinae); Ward, Blaimer & Fisher
2016 (Formicinae); Ward, Brady, Fisher & Schultz 2010 (Dolichoderinae); Borowiec 2016
(Dorylinae, no tribes recognised); Schmidt & Shattuck 2014 (Ponerinae); Ward & Fisher 2016
(Amblyoponinae, Fulakora resurrected); Camacho et al. 2022 (Ectatomminae); Griebenow 2024
(Leptanillinae).

Two placements are genuinely unsettled and are flagged inline: `apomyrma` (kept as Apomyrminae
per AntCat, though Ward & Fisher 2016 recover it inside Amblyoponinae) and `martialis`
(Martialinae still valid; the 2024 Leptanillinae revision does not touch it). `chalepoxenus` is
a junior synonym of Temnothorax and will not resolve in a current catalogue.

ECOLOGY is deliberately sparse -- only genus-level, uncontroversial guilds. Genera are omitted
where the trait is species-level (invasive/tramp status), contested, or where the genus spans
guilds. Where a genus fits two labels the morphologically dominant one is kept.
"""

GENUS_TO_SUBFAMILY = {
    "acanthognathus": "Myrmicinae",  # Attini (former Dacetini)
    "acanthomyrmex": "Myrmicinae",  # Crematogastrini
    "acanthostichus": "Dorylinae",  # ex-Cerapachyinae; Dorylinae has no tribes
    "acromyrmex": "Myrmicinae",  # Attini
    "acropyga": "Formicinae",  # Plagiolepidini
    "adelomyrmex": "Myrmicinae",  # Solenopsidini (Adelomyrmecini syn. 2015)
    "aenictus": "Dorylinae",  # ex-Aenictinae
    "amblyopone": "Amblyoponinae",  # Amblyoponini
    "ancyridris": "Myrmicinae",  # Crematogastrini (Ward et al. 2015)
    "aneuretus": "Aneuretinae",  # Aneuretini; monotypic relict subfamily
    "ankylomyrma": "Agroecomyrmecinae",  # Ankylomyrmini; MOVED out of Myrmicinae (Ward et al. 2015)
    "anochetus": "Ponerinae",  # Ponerini
    "aphaenogaster": "Myrmicinae",  # Stenammini (genus known to be non-monophyletic)
    "apomyrma": "Apomyrminae",  # Apomyrmini. AntCat/AntWeb keep the monotypic subfamily;
    # phylogenomics (Ward & Fisher 2016) puts it as sister to
    # the rest of Amblyoponinae, and some authors sink it there.
    "atta": "Myrmicinae",  # Attini
    "azteca": "Dolichoderinae",  # Leptomyrmecini
    "basiceros": "Myrmicinae",  # Attini (former Basicerotini)
    "bothriomyrmex": "Dolichoderinae",  # Bothriomyrmecini
    "brachymyrmex": "Formicinae",  # Myrmelachistini (moved out of Plagiolepidini, 2016)
    "brachyponera": "Ponerinae",  # Ponerini (split from Pachycondyla s.l.)
    "calyptomyrmex": "Myrmicinae",  # Crematogastrini
    "camponotus": "Formicinae",  # Camponotini
    "carebara": "Myrmicinae",  # Crematogastrini (NOT Solenopsidini)
    "cataglyphis": "Formicinae",  # Formicini
    "cataulacus": "Myrmicinae",  # Crematogastrini
    "centromyrmex": "Ponerinae",  # Ponerini
    "cephalotes": "Myrmicinae",  # Attini (former Cephalotini)
    "cerapachys": "Dorylinae",  # restricted sense after Borowiec 2016
    "chalepoxenus": "Myrmicinae",  # Crematogastrini; junior synonym of Temnothorax (Ward et al. 2015)
    "cheliomyrmex": "Dorylinae",  # ex-Ecitoninae
    "colobopsis": "Formicinae",  # Camponotini; resurrected from Camponotus (Ward et al. 2016)
    "crematogaster": "Myrmicinae",  # Crematogastrini
    "cryptopone": "Ponerinae",  # Ponerini
    "cyphomyrmex": "Myrmicinae",  # Attini
    "daceton": "Myrmicinae",  # Attini (former Dacetini)
    "diacamma": "Ponerinae",  # Ponerini
    "discothyrea": "Proceratiinae",  # Proceratiini
    "dolichoderus": "Dolichoderinae",  # Dolichoderini
    "dorylus": "Dorylinae",  # ex-Dorylinae s.str./Ecitoninae complex
    "dorymyrmex": "Dolichoderinae",  # Leptomyrmecini
    "eciton": "Dorylinae",  # ex-Ecitoninae
    "ectatomma": "Ectatomminae",  # Ectatommini
    "formica": "Formicinae",  # Formicini
    "fulakora": "Amblyoponinae",  # Amblyoponini; resurrected from Stigmatomma (Ward & Fisher 2016)
    "gigantiops": "Formicinae",  # Gigantiopini (monotypic)
    "gnamptogenys": "Ectatomminae",  # Ectatommini (several segregate genera revived, 2022)
    "harpegnathos": "Ponerinae",  # Ponerini
    "hypoponera": "Ponerinae",  # Ponerini
    "iridomyrmex": "Dolichoderinae",  # Leptomyrmecini
    "labidus": "Dorylinae",  # ex-Ecitoninae
    "lasius": "Formicinae",  # Lasiini
    "leptanilla": "Leptanillinae",  # Leptanillini (incl. Yavnella, Noonilla; Griebenow 2024)
    "leptogenys": "Ponerinae",  # Ponerini
    "leptomyrmex": "Dolichoderinae",  # Leptomyrmecini
    "linepithema": "Dolichoderinae",  # Leptomyrmecini
    "liometopum": "Dolichoderinae",  # Tapinomini
    "lordomyrma": "Myrmicinae",  # Crematogastrini
    "manica": "Myrmicinae",  # Myrmicini
    "martialis": "Martialinae",  # monotypic; sister to Leptanillinae ("leptanillomorphs").
    # Still a valid separate subfamily in AntCat; Griebenow's
    # 2024 Leptanillinae revision did NOT synonymise it.
    "megalomyrmex": "Myrmicinae",  # Solenopsidini
    "melissotarsus": "Myrmicinae",  # Crematogastrini (Melissotarsini syn. 2015)
    "meranoplus": "Myrmicinae",  # Crematogastrini
    "messor": "Myrmicinae",  # Stenammini
    "monomorium": "Myrmicinae",  # Solenopsidini
    "myrmecia": "Myrmeciinae",  # Myrmeciini
    "myrmica": "Myrmicinae",  # Myrmicini
    "myrmicaria": "Myrmicinae",  # Solenopsidini (Myrmicariini syn. 2015) - NOT Crematogastrini
    "myrmoteras": "Formicinae",  # Myrmoteratini
    "mystrium": "Amblyoponinae",  # Amblyoponini
    "neivamyrmex": "Dorylinae",  # ex-Ecitoninae
    "neoponera": "Ponerinae",  # Ponerini (split from Pachycondyla s.l.)
    "nomamyrmex": "Dorylinae",  # ex-Ecitoninae
    "notoncus": "Formicinae",  # Melophorini (moved from Melophorini-outgroups, 2016)
    "nothomyrmecia": "Myrmeciinae",  # Prionomyrmecini (ex-Nothomyrmeciinae)
    "novomessor": "Myrmicinae",  # Stenammini; revived from Aphaenogaster (Demarco & Cognato 2015)
    "nylanderia": "Formicinae",  # Lasiini (moved from Plagiolepidini, 2016)
    "ochetellus": "Dolichoderinae",  # Leptomyrmecini
    "odontomachus": "Ponerinae",  # Ponerini
    "odontoponera": "Ponerinae",  # Ponerini
    "oecophylla": "Formicinae",  # Oecophyllini
    "onychomyrmex": "Amblyoponinae",  # Amblyoponini
    "ophthalmopone": "Ponerinae",  # Ponerini
    "pachycondyla": "Ponerinae",  # Ponerini (s.str. after Schmidt & Shattuck 2014)
    "paraponera": "Paraponerinae",  # Paraponerini; monotypic subfamily
    "paratrechina": "Formicinae",  # Lasiini (moved from Plagiolepidini, 2016)
    "parvaponera": "Ponerinae",  # Ponerini (split from Pachycondyla s.l.)
    "pheidole": "Myrmicinae",  # Attini (Pheidolini syn. 2015) - not a fungus-grower
    "plagiolepis": "Formicinae",  # Plagiolepidini
    "platythyrea": "Ponerinae",  # Platythyreini
    "plectroctena": "Ponerinae",  # Ponerini
    "pogonomyrmex": "Myrmicinae",  # Pogonomyrmecini
    "polyrhachis": "Formicinae",  # Camponotini
    "ponera": "Ponerinae",  # Ponerini
    "prionopelta": "Amblyoponinae",  # Amblyoponini
    "pristomyrmex": "Myrmicinae",  # Crematogastrini
    "proatta": "Myrmicinae",  # Crematogastrini (Proattini syn. 2015)
    "proceratium": "Proceratiinae",  # Proceratiini
    "prolasius": "Formicinae",  # Melophorini
    "protanilla": "Leptanillinae",  # Leptanillini (incl. Anomalomyrma; Griebenow 2024)
    "pseudomyrmex": "Pseudomyrmecinae",  # Pseudomyrmecini
    "rhytidoponera": "Ectatomminae",  # Ectatommini
    "santschiella": "Formicinae",  # Santschiellini (resurrected 2016; NOT Gesomyrmecini)
    "simopone": "Dorylinae",  # ex-Cerapachyinae
    "solenopsis": "Myrmicinae",  # Solenopsidini
    "sphinctomyrmex": "Dorylinae",  # restricted to Neotropics after Borowiec 2016
    "stigmatomma": "Amblyoponinae",  # Amblyoponini
    "streblognathus": "Ponerinae",  # Ponerini
    "strumigenys": "Myrmicinae",  # Attini (former Dacetini)
    "tapinoma": "Dolichoderinae",  # Tapinomini
    "tatuidris": "Agroecomyrmecinae",  # Agroecomyrmecini
    "technomyrmex": "Dolichoderinae",  # Tapinomini
    "terataner": "Myrmicinae",  # Crematogastrini
    "tetramorium": "Myrmicinae",  # Crematogastrini (incl. Anergates, Teleutomyrmex)
    "tetraponera": "Pseudomyrmecinae",  # Pseudomyrmecini
    "thaumatomyrmex": "Ponerinae",  # Ponerini (Thaumatomyrmecini syn. 2014)
    "tetheamyrma": "Myrmicinae",  # Crematogastrini (possibly sister to rest, weakly supported)
    "trachymyrmex": "Myrmicinae",  # Attini
    "typhlomyrmex": "Ectatomminae",  # Ectatommini (Typhlomyrmecini merged, Camacho et al. 2022)
    "vollenhovia": "Myrmicinae",  # Crematogastrini
    "wasmannia": "Myrmicinae",  # Attini (Blepharidattina) - not a fungus-grower
    "zasphinctus": "Dorylinae",  # split from Sphinctomyrmex (Borowiec 2016)
}


# Coarse ecological / functional guild. Deliberately sparse: only genera whose
# label is uncontroversial at the GENUS level are included. Genera omitted on
# purpose because the trait is species-level or contested: camponotus, crematogaster
# (mostly but not wholly arboreal), terataner, melissotarsus, harpegnathos (jumping,
# not a latch-mediated trap-jaw), thaumatomyrmex / plectroctena / centromyrmex
# (diet specialists, but not a standard guild), onychomyrmex and leptanilla
# (legionary/group-raiding but not true army ants), novomessor (not a true harvester),
# and all "invasive/tramp" labels (a species-level, not genus-level, property).
# Note where a genus fits two labels the morphologically dominant one is kept
# (e.g. daceton and acanthognathus are arboreal, but listed as trap-jaw).
GENUS_TO_ECOLOGY = {
    # true army ants: obligate collective raiding + nomadism + dichthadiiform queens
    "dorylus": "army ant",
    "eciton": "army ant",
    "labidus": "army ant",
    "aenictus": "army ant",
    "neivamyrmex": "army ant",
    "nomamyrmex": "army ant",
    "cheliomyrmex": "army ant",
    # attine fungiculture (atta + acromyrmex are the true leafcutters)
    "atta": "leafcutter/fungus-grower",
    "acromyrmex": "leafcutter/fungus-grower",
    "trachymyrmex": "leafcutter/fungus-grower",
    "cyphomyrmex": "leafcutter/fungus-grower",
    # power-amplified mandible strike
    "odontomachus": "trap-jaw",
    "anochetus": "trap-jaw",
    "strumigenys": "trap-jaw",
    "daceton": "trap-jaw",
    "acanthognathus": "trap-jaw",
    "myrmoteras": "trap-jaw",
    "mystrium": "trap-jaw",  # "snap-jaw": mandible-on-mandible loading, a
    # convergent mechanism distinct from Odontomachus
    # predominantly arboreal / plant-nesting at genus level
    "cephalotes": "arboreal",
    "pseudomyrmex": "arboreal",
    "tetraponera": "arboreal",
    "dolichoderus": "arboreal",
    "polyrhachis": "arboreal",
    "oecophylla": "arboreal",  # weaver ants (larval-silk nests)
    "azteca": "arboreal",
    "cataulacus": "arboreal",
    "colobopsis": "arboreal",  # phragmotic soldiers
    # granivory as the dominant food source
    "messor": "harvester",
    "pogonomyrmex": "harvester",
    # obligate hypogaeic, blind, depigmented workers
    "leptanilla": "subterranean/hypogaeic",
    "protanilla": "subterranean/hypogaeic",
    "martialis": "subterranean/hypogaeic",
    "apomyrma": "subterranean/hypogaeic",
    # obligate dulotic (slave-making) social parasite
    "chalepoxenus": "social parasite",
}

# aliases used by the analysis modules
SUBFAMILY = GENUS_TO_SUBFAMILY
ECOLOGY = GENUS_TO_ECOLOGY
