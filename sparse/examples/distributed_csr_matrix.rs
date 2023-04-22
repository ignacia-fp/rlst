use rlst_sparse::{
    traits::{
        index_layout::IndexLayout,
        indexable_vector::{IndexableVector, IndexableVectorView, IndexableVectorViewMut},
    },
    vector::DefaultMpiVector,
};

fn main() {
    pub use mpi::traits::*;
    pub use rlst_sparse::ghost_communicator::GhostCommunicator;
    pub use rlst_sparse::index_layout::DefaultMpiIndexLayout;
    pub use rlst_sparse::sparse::csr_mat::CsrMatrix;
    pub use rlst_sparse::sparse::mpi_csr_mat::MpiCsrMatrix;
    pub use rlst_sparse::traits::index_layout;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let n_domain = 6;
    let n_range = 2;

    let values = vec![1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0];
    let indices = vec![0, 1, 2, 5, 0, 1, 2];
    let indptr = vec![0, 4, 7];

    let csr_mat: Option<CsrMatrix<f64>>;
    if rank == 0 {
        csr_mat = Some(CsrMatrix::new((2, 6), indices, indptr, values));
    } else {
        csr_mat = None;
    }

    let domain_layout = DefaultMpiIndexLayout::new(n_domain, &world);
    let range_layout = DefaultMpiIndexLayout::new(n_range, &world);

    let dist_mat = MpiCsrMatrix::from_csr(csr_mat, &domain_layout, &range_layout, &world);
    let mut distributed_vec = DefaultMpiVector::<f64, _>::new(&domain_layout);
    let mut result_vec = DefaultMpiVector::<f64, _>::new(&range_layout);

    let mut vec_mut_view = distributed_vec.view_mut().unwrap();

    for (index, value) in (domain_layout.local_range().0..domain_layout.local_range().1).enumerate()
    {
        *vec_mut_view.get_mut(index).unwrap() = value as f64;
    }

    dist_mat.matmul(1.0, &distributed_vec, 1.0, &mut result_vec);

    if rank == 1 {
        println!("Result: {:#?}", result_vec.view().unwrap().data());
    }
}