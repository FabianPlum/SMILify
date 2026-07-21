"""Shared mutable module state (Transformation PCA components)."""

# Transformation PCA components computed by 'Load all unposed registered
# meshes'; consumed when exporting a model. Shared across operators.
computed_scaledirs = None
computed_transdirs = None


def clear_morph_pca_globals():
    """Clear the global morph PCA variables"""
    global computed_scaledirs, computed_transdirs
    computed_scaledirs = None
    computed_transdirs = None
    print("Cleared global morph PCA variables")


def get_morph_pca_status():
    """Check if Transformation PCA components are available"""
    global computed_scaledirs, computed_transdirs
    if computed_scaledirs is not None and computed_transdirs is not None:
        return True, f"Available - scaledirs: {computed_scaledirs.shape}, transdirs: {computed_transdirs.shape}"
    else:
        return False, "Not available - run 'Load all unposed registered meshes' first"
